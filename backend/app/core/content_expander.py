"""
Content Expander for enhancing retrieved documents with related code context.

This module provides post-retrieval content expansion by analyzing code relationships
like inheritance, composition, method calls, and parent-child relationships using
the relationship code_graph from Enhanced Unified Graph Builder.
"""

import logging
import os
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContentExpander:
    """
    Expands retrieved documents with related code context using code_graph traversal.

    Key features:
    - Expands small symbols (variables, fields) to their parent containers (methods, classes)
    - Analyzes inheritance and composition relationships
    - Intelligent deduplication (prefers classes over individual methods)
    - Code-focused filtering for better retrieval quality
    - Filters output to only architectural symbols (no methods, constructors, fields)
    """

    # Architectural symbols - output of expansion should only contain these types
    # Small symbols (methods, constructors, fields, variables) are used INTERNALLY
    # for graph traversal but filtered from the output
    # Must match graph_builder.ARCHITECTURAL_SYMBOLS and section_based_generator.ARCHITECTURAL_SYMBOL_TYPES
    ARCHITECTURAL_SYMBOL_TYPES = {
        # Top-level code symbols
        "class",
        "interface",
        "struct",
        "enum",
        "trait",
        "protocol",
        "function",  # Standalone functions only - methods excluded
        "constant",  # Module-level constants
        "macro",  # C/C++ preprocessor definitions
        "type_alias",  # using/typedef type aliases (C++, Rust, etc.)
        # Documentation symbols
        "markdown_document",
        "toml_document",
        "text_chunk",
        "text_document",
        "restructuredtext_document",
        "plaintext_document",
        "documentation",
    }

    def __init__(self, graph_store=None, graph_text_index=None):
        """
        Initialize content expander.

        Args:
            graph_store: NetworkX code_graph from Enhanced Unified Graph Builder
                        Should have ._node_index already built by graph builder
            graph_text_index: Optional GraphTextIndex (FTS5) instance for fuzzy
                            symbol resolution when all index strategies fail.
        """
        self.graph = graph_store
        self.graph_text_index = graph_text_index
        self.logger = logging.getLogger(__name__)

        # Node index for O(1) lookups: (symbol_name, file_path, language) -> node_id
        # Prefer using graph's pre-built index if available, otherwise build on first use
        self._node_index = None
        self._simple_name_index = None
        self._full_name_index = None
        self._name_index = None
        self._suffix_index = None
        self._decl_impl_index = None
        self._constant_def_index = None

        # Optional profiling stats (enabled via WIKIS_PROFILE_EXPANSION=1)
        self._profile_stats: dict[str, int] = {}

    @staticmethod
    def _simple_symbol_name(name: str) -> str:
        if not name:
            return ""
        # Handles qualified names like pkg.mod.Class.method or ns::Type
        return name.split(".")[-1].split("::")[-1]

    def _build_node_index(self):
        """
        Build index mapping (symbol_name, file_path, language) -> node_id for O(1) lookups.

        This replaces O(N) scans through all graph nodes. Built lazily on first use.
        Handles multiple nodes with same key by preferring actual definitions over import aliases.

        Key selection priority:
        1. Skip import aliases (nodes with './' in ID)
        2. Prefer actual definitions from the symbol's original file
        3. For duplicates, prefer nodes with matching file paths in node_id
        """
        if not self.graph:
            self._node_index = {}
            return

        profile = os.getenv("WIKIS_PROFILE_EXPANSION") == "1"
        start = time.perf_counter() if profile else 0.0

        self.logger.debug("Building node index for O(1) lookups...")
        index = {}

        for node_id in self.graph.nodes():
            # Skip import aliases (e.g., typescript::components::./types::User)
            # These are references, not definitions
            if "./" in node_id or "../" in node_id:
                continue

            node_data = self.graph.nodes[node_id]
            symbol_name = node_data.get("symbol_name", "")
            file_path = node_data.get("file_path", "")
            language = node_data.get("language", "")
            symbol_type = node_data.get("symbol_type", "").lower()

            if not symbol_name or not file_path:
                continue

            # Create index keys.
            # We store both the raw symbol_name and its simple form to avoid index misses
            # when docs use qualified names but the graph stores local names (or vice versa).
            symbol_name_simple = self._simple_symbol_name(symbol_name)
            keys = {(symbol_name, file_path, language)}
            if symbol_name_simple and symbol_name_simple != symbol_name:
                keys.add((symbol_name_simple, file_path, language))

            # Handle duplicates: prefer actual definitions
            for key in keys:
                if key in index:
                    existing_node_id = index[key]
                    existing_data = self.graph.nodes[existing_node_id]
                    existing_type = existing_data.get("symbol_type", "").lower()

                    # Prefer architectural types
                    architectural_types = {"class", "interface", "struct", "enum", "function", "method"}

                    if symbol_type in architectural_types and existing_type not in architectural_types:
                        index[key] = node_id
                    elif symbol_type not in architectural_types and existing_type in architectural_types:
                        continue  # Keep existing architectural symbol
                    else:
                        # Both architectural: prefer node where file path appears in node_id
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        if f"::{file_name}::" in node_id and f"::{file_name}::" not in existing_node_id:
                            index[key] = node_id
                else:
                    index[key] = node_id

        self._node_index = index
        # Persist to graph for reuse across components (and across multiple ContentExpander instances)
        try:
            self.graph._node_index = index  # type: ignore[attr-defined]
        except Exception:  # noqa: S110
            pass

        if profile:
            elapsed = time.perf_counter() - start
            self.logger.info(f"[EXPAND_PROFILE] Built node index: {len(index)} entries in {elapsed:.2f}s")
        else:
            self.logger.debug(f"Built node index with {len(index)} entries")

    def expand_retrieved_documents(self, documents: list[Document], apply_code_filters: bool = True) -> list[Document]:
        """
        SIMPLIFIED: Expand retrieved architectural documents with related context.

        Since vector store only contains architectural symbols, all retrieved documents
        are already architectural symbols with chunk_type='symbol' - no filtering needed.

        Args:
            documents: List of retrieved documents (already architectural symbols only)
            apply_code_filters: Kept for compatibility, but only does simple deduplication

        Returns:
            List of expanded documents with related context
        """
        if not documents:
            return documents

        profile = os.getenv("WIKIS_PROFILE_EXPANSION") == "1"
        start = time.perf_counter() if profile else 0.0

        try:
            # Since all documents are already architectural symbols, just expand them
            all_docs = documents.copy()  # Start with originals
            processed_nodes = set()

            # Create initial symbol tracking from retrieved documents to enforce uniqueness
            existing_symbols = self._build_existing_symbol_index(documents)

            # Expand each document with comprehensive context
            for i, doc in enumerate(documents):
                expanded_context = self._expand_document_comprehensively(doc, processed_nodes, existing_symbols)

                # Check if any expanded doc is marked as a replacement for the original
                # (e.g., augmented C++ struct with implementations)
                replacement_doc = None
                other_context = []
                for context_doc in expanded_context:
                    if context_doc.metadata.get("replaces_original") and context_doc.metadata.get(
                        "symbol_name"
                    ) == doc.metadata.get("symbol_name"):
                        replacement_doc = context_doc
                    else:
                        other_context.append(context_doc)

                # If we have a replacement, use it instead of the original
                if replacement_doc:
                    all_docs[i] = replacement_doc
                    self.logger.debug(
                        f"Replaced original doc with augmented version for {doc.metadata.get('symbol_name')}"
                    )

                # Add other context documents
                all_docs.extend(other_context)

            # Simple deduplication only (should be minimal now with uniqueness checking)
            deduped_docs = self._simple_deduplicate(all_docs) if apply_code_filters else all_docs

            # Filter to only architectural symbols - small symbols (methods, constructors, fields)
            # were used internally for graph traversal but should not be in final output
            final_docs = self._filter_to_architectural_symbols(deduped_docs)

            if profile:
                elapsed = time.perf_counter() - start
                self.logger.info(
                    f"[EXPAND_PROFILE] Expansion total: {len(documents)} -> {len(final_docs)} docs in {elapsed:.2f}s"
                )
                if self._profile_stats:
                    stats_str = ", ".join(f"{k}={v}" for k, v in sorted(self._profile_stats.items()))
                    self.logger.info(f"[EXPAND_PROFILE] Node lookup stats: {stats_str}")
            else:
                self.logger.info(f"Simplified architectural expansion: {len(documents)} -> {len(final_docs)} documents")
            return final_docs

        except Exception as e:
            self.logger.error(f"Error during content expansion: {e}")
            return documents

    def _is_code_document(self, doc: Document) -> bool:
        """
        Check if document is a code symbol using chunk_type.
        Vector store contains only architectural symbols with chunk_type='symbol'.
        """
        chunk_type = doc.metadata.get("chunk_type", "").lower()
        return chunk_type == "symbol"

    def _expand_document_comprehensively(
        self, doc: Document, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand a single code document with comprehensive context for best-in-class documentation.

        Important: Does NOT include the original document - only adds expanded context.
        The original document is already in the retrieved set, so we only add related context.

        For each symbol type, provides the most complete context:
        - Classes: Include inheritance hierarchy, constructors, key methods
        - Methods/Functions: Include parent class, called functions, parameters
        - Variables/Fields: Include containing class/method context
        - Small symbols: Expand to meaningful parents

        Args:
            doc: Document to expand
            processed_nodes: Set of already processed code_graph nodes to avoid duplicates
            existing_symbols: Set of symbols already in retrieved documents to avoid duplicates

        Returns:
            List of documents with expanded context (NOT including the original)
        """
        result_docs = []

        if not self.graph:
            return result_docs

        # Extract symbol information from document metadata
        symbol_info = self._extract_symbol_info_from_document(doc)
        if not symbol_info:
            return result_docs

        symbol_name, symbol_type, file_path, language = symbol_info

        # Find the code_graph node for this symbol
        graph_node = self._find_graph_node(symbol_name, file_path, language)
        if not graph_node or graph_node in processed_nodes:
            return result_docs

        processed_nodes.add(graph_node)

        # Comprehensive expansion based on symbol type
        if symbol_type in ["class", "interface", "struct"]:
            # For classes/interfaces/structs: provide architectural context (inheritance, composition)
            # NOT member expansion since classes already contain all their members
            if symbol_type in ["struct", "class"] and language == "cpp":
                # C++ structs/classes get enhanced expansion with method implementations
                # Pass original doc to preserve is_initially_retrieved flag
                self.logger.debug(f"Calling _expand_cpp_struct_comprehensively for {symbol_name}")
                context_docs = self._expand_cpp_struct_comprehensively(
                    graph_node, processed_nodes, existing_symbols, original_doc=doc
                )
                self.logger.debug(f"_expand_cpp_struct_comprehensively returned {len(context_docs)} documents")
            elif symbol_type in ["struct", "class"] and language == "go":
                # Go structs get enhanced expansion with cross-file receiver methods
                # Go's receiver pattern means methods are top-level decls, not nested
                self.logger.debug(f"Calling _expand_go_struct_comprehensively for {symbol_name}")
                context_docs = self._expand_go_struct_comprehensively(
                    graph_node, processed_nodes, existing_symbols, original_doc=doc
                )
                self.logger.debug(f"_expand_go_struct_comprehensively returned {len(context_docs)} documents")
            else:
                # Standard class/interface expansion (for other languages or interfaces)
                self.logger.debug(f"Calling _expand_class_comprehensively for {symbol_name}")
                context_docs = self._expand_class_comprehensively(graph_node, processed_nodes, existing_symbols)
                self.logger.debug(f"_expand_class_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)

        elif symbol_type in ["method", "function"]:
            # For methods/functions: include parent class and related functions
            self.logger.debug(f"Calling _expand_method_comprehensively for {symbol_name}")
            context_docs = self._expand_method_comprehensively(graph_node, processed_nodes, doc, existing_symbols)
            self.logger.debug(f"_expand_method_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)

        elif symbol_type in ["constructor", "constructor_call"]:
            # For constructors and constructor calls: expand appropriately
            self.logger.debug(f"Calling _expand_constructor_comprehensively for {symbol_name}")
            context_docs = self._expand_constructor_comprehensively(graph_node, processed_nodes, existing_symbols)
            self.logger.debug(f"_expand_constructor_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)

        elif symbol_type == "constant" and language == "cpp":
            # C++ constants: augment extern declarations with actual values from implementations
            self.logger.debug(f"Handling C++ constant {symbol_name}")
            augmented_doc = self._augment_cpp_constant_with_definition(graph_node, doc)
            if augmented_doc:
                result_docs.append(augmented_doc)
                self.logger.debug(f"Augmented C++ constant {symbol_name} with definition")

        elif symbol_type == "constant":
            # Constants (all languages): expand to initialization functions/types
            # Strategy 3: const DEFAULT_USER = createDefaultUser() → bring createDefaultUser into context
            self.logger.debug(f"Expanding constant {symbol_name}")
            context_docs = self._expand_constant_comprehensively(graph_node, processed_nodes, existing_symbols)
            self.logger.debug(f"_expand_constant_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)

        elif symbol_type in ["variable", "field", "property"]:
            # For small symbols: expand to meaningful containing context
            context_docs = self._expand_small_symbol_comprehensively(graph_node, processed_nodes, existing_symbols)
            result_docs.extend(context_docs)

        elif symbol_type in ["enum", "annotation"]:
            # For type definitions: include usage context
            context_docs = self._expand_type_definition_comprehensively(graph_node, processed_nodes, existing_symbols)
            result_docs.extend(context_docs)

        elif symbol_type == "type_alias":
            # For type aliases: follow ALIAS_OF chain to bring underlying types into context
            # e.g. using MyVec = vector<MyClass> → bring MyClass definition
            self.logger.debug(f"Expanding type alias {symbol_name}")
            context_docs = self._expand_type_alias_comprehensively(graph_node, processed_nodes, existing_symbols)
            self.logger.debug(f"_expand_type_alias_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)

        return result_docs

    def _extract_symbol_info_from_document(self, doc: Document) -> tuple[str, str, str, str] | None:
        """Extract symbol information from document metadata."""
        metadata = doc.metadata

        # Try different metadata formats
        symbol_name = metadata.get("symbol_name") or metadata.get("name") or metadata.get("title", "").split("/")[-1]

        symbol_type = metadata.get("symbol_type") or metadata.get("type", "").lower()

        file_path = metadata.get("file_path") or metadata.get("source", "")

        language = metadata.get("language") or metadata.get("lang", "")

        if symbol_name and file_path:
            return symbol_name, symbol_type, file_path, language

        return None

    def _find_graph_node(self, symbol_name: str, file_path: str, language: str) -> str | None:
        """
        Find code_graph node for a symbol using O(1) index lookup.

        Replaces O(N) scan with indexed lookup for performance.
        Falls back to partial matching if exact lookup fails.

        Args:
            symbol_name: Name of the symbol
            file_path: File path where symbol is defined
            language: Programming language

        Returns:
            Node ID if found, None otherwise
        """
        if not self.graph:
            return None

        self._ensure_graph_indexes()

        profile = os.getenv("WIKIS_PROFILE_EXPANSION") == "1"

        # Extract simple name and file name for matching
        simple_name = self._simple_symbol_name(symbol_name)
        os.path.splitext(os.path.basename(file_path))[0] if file_path else ""

        self.logger.debug(f"Looking for node: symbol='{symbol_name}', file='{file_path}', language='{language}'")

        # Try exact lookup first (O(1))
        key = (symbol_name, file_path, language)
        if key in self._node_index:
            node_id = self._node_index[key]
            if profile:
                self._profile_stats["node_index_hit_exact"] = self._profile_stats.get("node_index_hit_exact", 0) + 1
            self.logger.debug(f"Found via exact index lookup: {node_id}")
            return node_id

        # Try with simple name
        key = (simple_name, file_path, language)
        if self._simple_name_index and key in self._simple_name_index:
            node_id = self._simple_name_index[key]
            if profile:
                self._profile_stats["node_index_hit_simple"] = self._profile_stats.get("node_index_hit_simple", 0) + 1
            self.logger.debug(f"Found via simple name index lookup: {node_id}")
            return node_id

        # Try full name index (best-effort)
        full_name_candidates = []
        if symbol_name and "." in symbol_name:
            full_name_candidates.append(symbol_name)
        if symbol_name and "::" in symbol_name:
            full_name_candidates.append(symbol_name.replace("::", "."))
        for full_name in full_name_candidates:
            if self._full_name_index and full_name in self._full_name_index:
                node_id = self._full_name_index[full_name]
                if profile:
                    self._profile_stats["node_index_hit_full_name"] = (
                        self._profile_stats.get("node_index_hit_full_name", 0) + 1
                    )
                self.logger.debug(f"Found via full name index lookup: {node_id}")
                return node_id

        # Try suffix index (qualified suffix match)
        def _select_candidate(candidates):
            if not candidates:
                return None
            # Prefer exact file+language match
            for candidate in candidates:
                node_data = self.graph.nodes.get(candidate, {})
                if node_data.get("file_path") == file_path and node_data.get("language") == language:
                    return candidate
            # Prefer file match
            for candidate in candidates:
                node_data = self.graph.nodes.get(candidate, {})
                if node_data.get("file_path") == file_path:
                    return candidate
            # Prefer language match
            for candidate in candidates:
                node_data = self.graph.nodes.get(candidate, {})
                if node_data.get("language") == language:
                    return candidate
            return candidates[0]

        for suffix_key in [symbol_name, simple_name]:
            if suffix_key and self._suffix_index and suffix_key in self._suffix_index:
                node_id = _select_candidate(self._suffix_index[suffix_key])
                if node_id:
                    if profile:
                        self._profile_stats["node_index_hit_suffix"] = (
                            self._profile_stats.get("node_index_hit_suffix", 0) + 1
                        )
                    self.logger.debug(f"Found via suffix index lookup: {node_id}")
                    return node_id

        # Try name index (priority-ordered list)
        if symbol_name and self._name_index and symbol_name in self._name_index:
            node_id = _select_candidate(self._name_index[symbol_name])
            if node_id:
                if profile:
                    self._profile_stats["node_index_hit_name"] = self._profile_stats.get("node_index_hit_name", 0) + 1
                self.logger.debug(f"Found via name index lookup: {node_id}")
                return node_id

        # Strategy 6: FTS5 fuzzy search (safety net for unresolved symbols)
        # When all O(1) index strategies fail, use FTS5 BM25 search to find
        # an approximate match.  This handles typos, partial names, and
        # renamed symbols that haven't been re-indexed yet.
        if (
            self.graph_text_index is not None
            and hasattr(self.graph_text_index, "is_open")
            and self.graph_text_index.is_open
        ):
            try:
                fts_docs = self.graph_text_index.search_smart(
                    symbol_name,
                    k=3,
                    intent="symbol",
                )
                if fts_docs:
                    # Pick the best match, preferring same file and language
                    best_node = None
                    for doc in fts_docs:
                        doc_sym = doc.metadata.get("symbol_name", "")
                        doc_path = doc.metadata.get("rel_path", "")
                        # Build candidate node ID (same format as graph)
                        candidate = f"{doc_path}::{doc_sym}" if doc_path else doc_sym
                        if candidate in self.graph:
                            if not best_node:
                                best_node = candidate
                            # Prefer same-file match
                            if file_path and doc_path and file_path.endswith(doc_path):
                                best_node = candidate
                                break
                    if best_node:
                        if profile:
                            self._profile_stats["node_index_hit_fts5"] = (
                                self._profile_stats.get("node_index_hit_fts5", 0) + 1
                            )
                        self.logger.debug(f"Found via FTS5 fuzzy search: {best_node}")
                        return best_node
            except Exception as e:
                self.logger.debug(f"FTS5 fuzzy search failed for '{symbol_name}': {e}")

        if profile:
            self._profile_stats["node_index_miss"] = self._profile_stats.get("node_index_miss", 0) + 1
        self.logger.debug(f"Index lookup failed for symbol: {symbol_name}")
        return None

    def _ensure_graph_indexes(self) -> None:
        """Ensure local references to graph indexes are initialized."""
        if not self.graph:
            return

        if self._node_index is None:
            if hasattr(self.graph, "_node_index") and self.graph._node_index:
                self._node_index = self.graph._node_index
                self.logger.debug(f"Reusing graph builder's node index ({len(self._node_index)} entries)")
            else:
                self._build_node_index()

        if self._simple_name_index is None:
            self._simple_name_index = getattr(self.graph, "_simple_name_index", {})
        if self._full_name_index is None:
            self._full_name_index = getattr(self.graph, "_full_name_index", {})
        if self._name_index is None:
            self._name_index = getattr(self.graph, "_name_index", {})
        if self._suffix_index is None:
            self._suffix_index = getattr(self.graph, "_suffix_index", {})
        if self._decl_impl_index is None:
            self._decl_impl_index = getattr(self.graph, "_decl_impl_index", {})
        if self._constant_def_index is None:
            self._constant_def_index = getattr(self.graph, "_constant_def_index", {})

    def _needs_parent_expansion(self, symbol_type: str) -> bool:
        """
        Check if a symbol type needs parent context expansion based on graph edges.

        Replaced hard-coded hierarchy with dynamic edge checking:
        - Complete containers (class, interface, struct, function, module): NO parent expansion
        - Members (method, field, variable, parameter): YES parent expansion (if DEFINES edge exists)

        This allows the graph structure to define relationships, not hard-coded rules.
        """
        # Complete architectural containers don't need parent expansion
        # They expand outward to architecture, not upward to parents
        complete_types = {"class", "interface", "struct", "function", "module", "namespace", "enum"}

        return symbol_type.lower() not in complete_types

    def _find_parent_context(self, graph_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Find parent context for a symbol using DEFINES edges (O(1) traversal).

        Simplified from 3-stage fallback to direct edge traversal:
        1. Check DEFINES edges (primary - local containment)
        2. Check parent_symbol metadata (fallback for legacy data)
        3. Check other containment edges (contains, has_member)

        Eliminates O(N) scan (old Strategy 3) by relying on graph edges.
        """
        parent_docs = []

        if not self.graph or not self.graph.has_node(graph_node):
            return parent_docs

        node_data = self.graph.nodes[graph_node]
        symbol_name = node_data.get("symbol_name", graph_node.split("::")[-1])

        # Priority 1: DEFINES edges (O(1) via graph.predecessors)
        # Example: Class.method has incoming DEFINES edge from Class
        for pred in self.graph.predecessors(graph_node):
            if pred in processed_nodes:
                continue

            edge_data = self.graph.get_edge_data(pred, graph_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type == "defines":  # Prioritize DEFINES
                        parent_doc = self._create_document_from_graph_node(
                            pred,
                            expansion_reason="defines",  # Parent defines child
                            source_symbol=symbol_name,
                        )
                        if parent_doc:
                            parent_docs.append(parent_doc)
                            processed_nodes.add(pred)
                            self.logger.debug(f"Found parent via DEFINES edge: {pred}")
                            return parent_docs  # Found via DEFINES, return immediately

        # Priority 2: parent_symbol metadata (fallback for legacy data)
        parent_symbol = node_data.get("parent_symbol")
        if parent_symbol:
            parent_node = self._find_graph_node(
                parent_symbol, node_data.get("file_path", ""), node_data.get("language", "")
            )
            if parent_node and parent_node not in processed_nodes:
                parent_doc = self._create_document_from_graph_node(
                    parent_node, expansion_reason="contains", source_symbol=symbol_name
                )
                if parent_doc:
                    parent_docs.append(parent_doc)
                    processed_nodes.add(parent_node)
                    self.logger.debug(f"Found parent via parent_symbol metadata: {parent_node}")
                    return parent_docs

        # Priority 3: Other containment edges (contains, has_member)
        for pred in self.graph.predecessors(graph_node):
            if pred in processed_nodes:
                continue

            edge_data = self.graph.get_edge_data(pred, graph_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type in ["contains", "has_member"]:
                        parent_doc = self._create_document_from_graph_node(
                            pred, expansion_reason=rel_type, source_symbol=symbol_name
                        )
                        if parent_doc:
                            parent_docs.append(parent_doc)
                            processed_nodes.add(pred)
                            self.logger.debug(f"Found parent via {rel_type} edge: {pred}")
                            break

        # Note: Removed O(N) Strategy 3 scan (_find_containing_symbol)
        # All containment should be captured by DEFINES/contains/has_member edges

        return parent_docs

    def _create_document_from_graph_node(
        self, node_id: str, expansion_reason: str = None, source_symbol: str = None, via_context: str = None
    ) -> Document | None:
        """
        Create a Document from a code_graph node.

        Args:
            node_id: Graph node ID to create document from
            expansion_reason: Why this doc was expanded (e.g., "inherited_by", "composes", "calls")
            source_symbol: The symbol that triggered this expansion (for backward hints)
            via_context: Transitive context like "via repo field" or "via getUser() method"
        """
        if not self.graph or not self.graph.has_node(node_id):
            logger.debug(f"Graph or node not found for {node_id}")
            return None

        node_data = self.graph.nodes[node_id]
        symbol = node_data.get("symbol")

        if symbol and hasattr(symbol, "source_text") and symbol.source_text:
            content = symbol.source_text
        else:
            # Fallback to node data or create placeholder content
            content = node_data.get("content")
            if not content:
                symbol_name = node_data.get("symbol_name", node_id)
                symbol_type = node_data.get("symbol_type", "symbol")
                content = f"Symbol: {symbol_name} ({symbol_type})"

        # Ensure content is not None
        if not content:
            content = f"Graph node: {node_id}"

        # Get relative path, fallback to file_path if not available
        rel_path = node_data.get("rel_path") or node_data.get("file_path", "")

        # Extract line numbers: try direct node attributes first (basic graph),
        # then fall back to the symbol object (comprehensive graph).
        start_line = node_data.get("start_line", 0)
        end_line = node_data.get("end_line", 0)
        if (not start_line or not end_line) and symbol:
            start_line = getattr(symbol, "start_line", 0) or 0
            end_line = getattr(symbol, "end_line", 0) or 0
            # Rich parser symbols store lines in Range object, not flat attrs
            if not start_line or not end_line:
                _range = getattr(symbol, "range", None)
                if _range:
                    start_line = getattr(_range.start, "line", 0) or 0
                    end_line = getattr(_range.end, "line", 0) or 0

        metadata = {
            "node_id": node_id,  # Include node_id for further expansion
            "symbol_name": node_data.get("symbol_name", ""),
            "symbol_type": node_data.get("symbol_type", ""),
            "file_path": node_data.get("file_path", ""),
            "rel_path": rel_path,
            "language": node_data.get("language", ""),
            "chunk_type": "symbol",
            "source": rel_path,  # Use relative path for source to avoid absolute paths in documentation
            "start_line": start_line,
            "end_line": end_line,
        }

        # Add expansion context for backward hints (← expanded docs tell WHY they were included)
        if expansion_reason:
            metadata["expansion_reason"] = expansion_reason
        if source_symbol:
            metadata["expansion_source"] = source_symbol
        if via_context:
            metadata["expansion_via"] = via_context

        logger.debug(
            f"Created document from node {node_id}: {metadata['symbol_name']} ({metadata['symbol_type']})"
            f"{f' [reason: {expansion_reason}]' if expansion_reason else ''}"
        )
        return Document(page_content=content, metadata=metadata)

    def _simple_deduplicate(self, documents: list[Document]) -> list[Document]:
        """
        Simple deduplication based on symbol name and file path.
        Since we only have architectural symbols, keep it simple.
        """
        seen = set()
        deduplicated = []

        for doc in documents:
            symbol_name = doc.metadata.get("symbol_name", "")
            file_path = doc.metadata.get("file_path", "")
            symbol_type = doc.metadata.get("symbol_type", "")

            # Create unique key
            key = f"{file_path}::{symbol_name}::{symbol_type}"

            if key not in seen:
                seen.add(key)
                deduplicated.append(doc)

        self.logger.debug(f"Simple deduplication: {len(documents)} -> {len(deduplicated)} documents")
        return deduplicated

    def _filter_to_architectural_symbols(self, documents: list[Document]) -> list[Document]:
        """
        Filter documents to only include architectural symbols.

        Small symbols (methods, constructors, fields, variables) are used INTERNALLY
        during graph traversal to find transitive relationships, but should not be
        included in the final output.

        Args:
            documents: All expanded documents (may include non-architectural symbols)

        Returns:
            List containing only architectural symbols
        """
        from collections import Counter

        # Count initial distribution by symbol type
        initial_distribution = Counter()
        for doc in documents:
            symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", "")).lower()
            chunk_type = doc.metadata.get("chunk_type", "").lower()
            if chunk_type in ["text_document", "text_chunk", "documentation"]:
                initial_distribution["[text_document]"] += 1
            elif symbol_type:
                initial_distribution[symbol_type] += 1
            else:
                initial_distribution["[unknown]"] += 1

        self.logger.info(f"[ARCH_FILTER] Input: {len(documents)} documents")
        self.logger.info(f"[ARCH_FILTER] Distribution before filter: {dict(initial_distribution)}")

        architectural_docs = []
        filtered_distribution = Counter()

        for doc in documents:
            symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", "")).lower()
            chunk_type = doc.metadata.get("chunk_type", "").lower()

            # Text documents (markdown, etc.) are always architectural
            if chunk_type in ["text_document", "text_chunk", "documentation"]:
                architectural_docs.append(doc)
                continue

            # Check if symbol type is architectural
            if symbol_type in self.ARCHITECTURAL_SYMBOL_TYPES:
                architectural_docs.append(doc)
            else:
                filtered_distribution[symbol_type] += 1
                self.logger.debug(
                    f"Filtered non-architectural: {symbol_type} | {doc.metadata.get('symbol_name', 'unknown')}"
                )

        # Log what was filtered
        if filtered_distribution:
            self.logger.info(f"[ARCH_FILTER] Filtered out: {dict(filtered_distribution)}")

        # Count final distribution
        final_distribution = Counter()
        for doc in architectural_docs:
            symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", "")).lower()
            chunk_type = doc.metadata.get("chunk_type", "").lower()
            if chunk_type in ["text_document", "text_chunk", "documentation"]:
                final_distribution["[text_document]"] += 1
            elif symbol_type:
                final_distribution[symbol_type] += 1
            else:
                final_distribution["[unknown]"] += 1

        self.logger.info(f"[ARCH_FILTER] Output: {len(architectural_docs)} documents")
        self.logger.info(f"[ARCH_FILTER] Distribution after filter: {dict(final_distribution)}")

        return architectural_docs

    def _build_existing_symbol_index(self, documents: list[Document]) -> set[str]:
        """
        Build an index of symbols that already exist in the retrieved documents.

        This ensures we don't add duplicates during expansion - we only add symbols
        that don't already exist in the original retrieved set.

        Args:
            documents: Original documents retrieved from vector store

        Returns:
            Set of unique keys for existing symbols
        """
        existing_symbols = set()

        for doc in documents:
            symbol_name = doc.metadata.get("symbol_name", "")
            file_path = doc.metadata.get("file_path", "")
            symbol_type = doc.metadata.get("symbol_type", "")

            # Create the same unique key format used in deduplication
            key = f"{file_path}::{symbol_name}::{symbol_type}"
            existing_symbols.add(key)

        self.logger.debug(f"Built existing symbol index with {len(existing_symbols)} symbols from retrieved documents")
        return existing_symbols

    def _is_symbol_already_retrieved(self, doc: Document, existing_symbols: set[str]) -> bool:
        """
        Check if a symbol already exists in the retrieved documents.

        Args:
            doc: Document to check
            existing_symbols: Set of unique keys for existing symbols

        Returns:
            True if symbol already exists in retrieved set, False otherwise
        """
        if not existing_symbols:
            return False

        symbol_name = doc.metadata.get("symbol_name", "")
        file_path = doc.metadata.get("file_path", "")
        symbol_type = doc.metadata.get("symbol_type", "")

        key = f"{file_path}::{symbol_name}::{symbol_type}"
        return key in existing_symbols

    def _filter_unique_documents(self, docs: list[Document], existing_symbols: set[str] = None) -> list[Document]:
        """
        Filter out documents that already exist in the retrieved set.

        Args:
            docs: List of documents to filter
            existing_symbols: Set of symbols already in retrieved documents

        Returns:
            List of documents that don't already exist in retrieved set
        """
        if not existing_symbols:
            return docs

        unique_docs = []
        for doc in docs:
            if not self._is_symbol_already_retrieved(doc, existing_symbols):
                unique_docs.append(doc)
            else:
                # Log when we skip a duplicate
                symbol_name = doc.metadata.get("symbol_name", "unknown")
                symbol_type = doc.metadata.get("symbol_type", "unknown")
                self.logger.debug(f"Skipping duplicate symbol from expansion: {symbol_name} ({symbol_type})")

        return unique_docs

    def _expand_class_comprehensively(
        self, class_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Provide comprehensive architectural context for a class.

        Classes are complete containers that already include all their members (methods, fields, constructors).
        Instead, we expand classes to provide broader architectural understanding:
        - Parent classes (inheritance hierarchy)
        - Implementing classes (for interfaces)
        - Related classes (composition relationships)

        This gives architectural context without duplicating what's already in the class.
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            self.logger.debug(f"Graph or node not found for {class_node}")
            return context_docs

        node_data = self.graph.nodes[class_node]
        class_name = node_data.get("symbol_name", "")
        symbol_type = node_data.get("symbol_type", "").lower()

        self.logger.debug(f"Expanding {symbol_type} {class_name} for architectural context")

        # 1. Find inheritance context (parent classes, implemented interfaces, implementing classes)
        inheritance_docs = self._find_inheritance_context(class_node, processed_nodes)
        # Filter out duplicates with retrieved documents
        inheritance_docs = self._filter_unique_documents(inheritance_docs, existing_symbols)
        self.logger.debug(f"Found {len(inheritance_docs)} unique inheritance/implementation docs")
        context_docs.extend(inheritance_docs)

        # 2. Find composition relationships (classes this class uses)
        composition_docs = self._find_composition_context(class_node, processed_nodes, limit=3)
        # Filter out duplicates with retrieved documents
        composition_docs = self._filter_unique_documents(composition_docs, existing_symbols)
        self.logger.debug(f"Found {len(composition_docs)} unique composition docs")
        context_docs.extend(composition_docs)

        # 2b. NEW: Find parameter type enums/classes used by methods of this class
        # This captures: PermissionChecker -> hasPermission(role: Role) -> Role enum
        param_type_docs = self._expand_class_method_parameter_types(class_node, processed_nodes, existing_symbols)
        param_type_docs = self._filter_unique_documents(param_type_docs, existing_symbols)
        self.logger.debug(f"Found {len(param_type_docs)} unique parameter type docs from class methods")
        context_docs.extend(param_type_docs)

        # 3. Find functions that use this class (parameters, return types)
        # Look at INCOMING relationships (predecessors) for functions that reference this class
        function_docs = self._find_functions_using_class(class_node, processed_nodes, limit=5)
        function_docs = self._filter_unique_documents(function_docs, existing_symbols)
        self.logger.debug(f"Found {len(function_docs)} unique functions using this class")
        context_docs.extend(function_docs)

        # 4. Find classes that use this class (composition/aggregation or method usage)
        # Look at INCOMING relationships (predecessors) for classes that reference this class
        # This captures the transitive relationship: Class A <- Method <- Class B => Class A relates to Class B
        using_class_docs = self._find_classes_using_class(class_node, processed_nodes, limit=5)
        using_class_docs = self._filter_unique_documents(using_class_docs, existing_symbols)
        self.logger.debug(f"Found {len(using_class_docs)} unique classes using this class")
        context_docs.extend(using_class_docs)

        # 5. NEW: Find free functions that this class calls (transitive call site expansion)
        # This captures: UserService --calls--> formatUserData (free function)
        # Strategy: Class methods call free functions → bring free functions into context
        called_function_docs = self._find_called_free_functions(class_node, processed_nodes, existing_symbols, limit=5)
        called_function_docs = self._filter_unique_documents(called_function_docs, existing_symbols)
        self.logger.debug(f"Found {len(called_function_docs)} unique called free functions")
        context_docs.extend(called_function_docs)

        self.logger.debug(
            f"{symbol_type.title()} {class_name} expanded with {len(context_docs)} architectural context documents"
        )
        return context_docs

    def _expand_cpp_struct_comprehensively(
        self,
        struct_node: str,
        processed_nodes: set[str],
        existing_symbols: set[str] = None,
        original_doc: Document = None,
    ) -> list[Document]:
        """
        Expand C++ struct with comprehensive context INCLUDING method implementations.

        C++ specific: Structs often have method declarations in .h and implementations in .cpp files.
        We augment the struct document with these implementations using DEFINES_BODY relationships.

        Args:
            struct_node: Graph node for the struct
            processed_nodes: Already processed nodes
            existing_symbols: Already retrieved symbols
            original_doc: Original document from retrieval (preserves is_initially_retrieved flag)

        Returns:
            List of expanded context documents (struct augmented with implementations + related classes)
        """
        context_docs = []

        # 1. Standard class expansion (inheritance, composition) - reuse existing logic
        standard_context = self._expand_class_comprehensively(struct_node, processed_nodes, existing_symbols)
        context_docs.extend(standard_context)

        # 2. C++ SPECIFIC: Augment struct document with method implementations from split files
        try:
            # Collect method implementations grouped by file
            impl_by_file = self._collect_cpp_method_implementations(struct_node, processed_nodes)

            if impl_by_file:
                # Use original document if provided (preserves is_initially_retrieved flag),
                # otherwise fall back to creating from graph node
                struct_doc = original_doc if original_doc else self._find_struct_document(struct_node)

                if struct_doc:
                    # Augment with implementations
                    augmented_doc = self._augment_struct_with_implementations(struct_doc, impl_by_file)

                    # Replace original struct with augmented version
                    context_docs = [
                        d
                        for d in context_docs
                        if d.metadata.get("symbol_name") != struct_doc.metadata.get("symbol_name")
                    ]
                    context_docs.insert(0, augmented_doc)

                    self.logger.info(f"Augmented C++ struct with {len(impl_by_file)} implementation files")
        except Exception as e:
            self.logger.error(f"Error augmenting C++ struct with implementations: {e}", exc_info=True)

        return context_docs

    def _expand_go_struct_comprehensively(
        self,
        struct_node: str,
        processed_nodes: set[str],
        existing_symbols: set[str] = None,
        original_doc: Document = None,
    ) -> list[Document]:
        """
        Expand Go struct with comprehensive context INCLUDING cross-file receiver methods.

        Go specific: Struct fields live in the struct body, but methods are top-level
        declarations with a receiver parameter — potentially in different files.
        We augment the struct document with these receiver methods using DEFINES edges.

        This is analogous to C++'s header/implementation split:
        - C++:  class Foo in foo.h + Foo::method() in foo.cpp → DEFINES + DEFINES_BODY
        - Go:   type Foo struct in foo.go + func (f *Foo) method() in foo_methods.go → DEFINES

        Args:
            struct_node: Graph node for the struct
            processed_nodes: Already processed nodes
            existing_symbols: Already retrieved symbols
            original_doc: Original document from retrieval (preserves is_initially_retrieved flag)

        Returns:
            List of expanded context documents (struct augmented with methods + related classes)
        """
        context_docs = []

        # 1. Standard class expansion (inheritance, composition) - reuse existing logic
        standard_context = self._expand_class_comprehensively(struct_node, processed_nodes, existing_symbols)
        context_docs.extend(standard_context)

        # 2. Go SPECIFIC: Augment struct document with cross-file receiver methods
        try:
            methods_by_file = self._collect_go_receiver_methods(struct_node, processed_nodes)

            if methods_by_file:
                struct_doc = original_doc if original_doc else self._find_struct_document(struct_node)

                if struct_doc:
                    augmented_doc = self._augment_struct_with_go_methods(struct_doc, methods_by_file)

                    # Replace original struct with augmented version
                    context_docs = [
                        d
                        for d in context_docs
                        if d.metadata.get("symbol_name") != struct_doc.metadata.get("symbol_name")
                    ]
                    context_docs.insert(0, augmented_doc)

                    total_methods = sum(len(m) for m in methods_by_file.values())
                    self.logger.info(
                        f"Augmented Go struct with {total_methods} receiver methods from {len(methods_by_file)} files"
                    )
        except Exception as e:
            self.logger.error(f"Error augmenting Go struct with receiver methods: {e}", exc_info=True)

        return context_docs

    def _collect_go_receiver_methods(self, struct_node: str, processed_nodes: set[str]) -> dict[str, list[str]]:
        """
        Collect Go receiver method source code grouped by file using DEFINES edges.

        In Go, methods are linked to structs via DEFINES edges with annotation
        ``cross_file: True`` when the method lives in a different file than the struct.

        Args:
            struct_node: Graph node for the struct
            processed_nodes: Already processed nodes (will be updated)

        Returns:
            Dict mapping rel_path -> list of method source texts
        """
        methods_by_file = {}

        if not self.graph or not self.graph.has_node(struct_node):
            return methods_by_file

        try:
            struct_data = self.graph.nodes.get(struct_node, {})
            struct_file = struct_data.get("rel_path", "") or struct_data.get("file_path", "")
            struct_name = struct_data.get("symbol_name", "")

            self.logger.debug(f"Collecting Go receiver methods for {struct_name} via DEFINES edges")

            for method_node in self.graph.successors(struct_node):
                edge_data = self.graph.get_edge_data(struct_node, method_node)
                if not edge_data:
                    continue

                is_defines = False
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type == "defines":
                        is_defines = True
                        break

                if not is_defines:
                    continue

                method_data = self.graph.nodes[method_node]
                symbol_type = method_data.get("symbol_type", "").lower()
                if symbol_type not in ["method", "function"]:
                    continue

                # Only collect cross-file methods (same-file are already in struct context)
                method_file = method_data.get("rel_path", "") or method_data.get("file_path", "")
                if not method_file or method_file == struct_file:
                    continue

                symbol_obj = method_data.get("symbol")
                if not symbol_obj or not hasattr(symbol_obj, "source_text"):
                    continue

                source_text = symbol_obj.source_text
                if not source_text or not source_text.strip():
                    continue

                if method_file not in methods_by_file:
                    methods_by_file[method_file] = []
                methods_by_file[method_file].append(source_text)
                processed_nodes.add(method_node)

                method_name = method_data.get("symbol_name", "unknown")
                self.logger.debug(f"Collected receiver method {method_name} from {method_file}")

        except Exception as e:
            self.logger.error(f"Error collecting Go receiver methods: {e}", exc_info=True)

        return methods_by_file

    def _augment_struct_with_go_methods(self, struct_doc: Document, methods_by_file: dict[str, list[str]]) -> Document:
        """
        Create an augmented Document with Go receiver methods appended.

        Args:
            struct_doc: Original struct document
            methods_by_file: Dict of rel_path -> list of method source texts

        Returns:
            New Document with augmented page_content
        """
        original_content = struct_doc.page_content
        augmented_content = original_content

        for method_file, method_texts in sorted(methods_by_file.items()):
            count = len(method_texts)
            augmented_content += (
                f"\n\n// Receiver methods from {method_file} ({count} method{'s' if count != 1 else ''})\n"
            )
            augmented_content += "\n\n".join(method_texts)

        augmented_metadata = dict(struct_doc.metadata)
        augmented_metadata["augmented_with_go_methods"] = True
        total = sum(len(m) for m in methods_by_file.values())
        augmented_metadata["go_receiver_method_count"] = total

        return Document(
            page_content=augmented_content,
            metadata=augmented_metadata,
        )

    def _collect_cpp_method_implementations(self, class_node: str, processed_nodes: set[str]) -> dict[str, list[str]]:
        """
        Collect C++ method implementation source code grouped by file using DEFINES edges.

        Uses DEFINES edges to find all methods/constructors defined by this class,
        then collects implementations that are in different files (out-of-line).

        Args:
            class_node: Graph node for the class/struct
            processed_nodes: Already processed nodes (will be updated)

        Returns:
            Dict mapping rel_path (relative file path) -> list of method source texts
        """
        impl_by_file = {}

        if not self.graph or not self.graph.has_node(class_node):
            return impl_by_file

        try:
            # Get class file path
            class_data = self.graph.nodes.get(class_node, {})
            class_file = class_data.get("file_path", "")
            class_name = class_data.get("symbol_name", "")

            self.logger.debug(f"Collecting C++ implementations for {class_name} using DEFINES/DEFINES_BODY edges")

            # Strategy 1: Use DEFINES edges to find all methods/constructors (inline and declarations)
            # Strategy 2: Follow DEFINES_BODY edges from declarations to find out-of-line implementations

            declarations = []  # Track declaration nodes to find their implementations

            for method_node in self.graph.successors(class_node):
                # Check if this is a DEFINES edge
                edge_data = self.graph.get_edge_data(class_node, method_node)
                if not edge_data:
                    continue

                # Check all edges between these nodes
                is_defines = False
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type == "defines":
                        is_defines = True
                        break

                if not is_defines:
                    continue

                # Check if it's a method or constructor
                method_data = self.graph.nodes[method_node]
                symbol_type = method_data.get("symbol_type", "").lower()

                if symbol_type not in ["method", "constructor"]:
                    continue

                declarations.append(method_node)

            self.logger.debug(f"Found {len(declarations)} method/constructor declarations via DEFINES")

            # Now follow DEFINES_BODY edges to find out-of-line implementations
            for decl_node in declarations:
                # Find all nodes that have DEFINES_BODY edge pointing to this declaration
                for impl_node in self.graph.predecessors(decl_node):
                    edge_data = self.graph.get_edge_data(impl_node, decl_node)
                    if not edge_data:
                        continue

                    # Check if this is a DEFINES_BODY edge (implementation → declaration)
                    is_defines_body = False
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type == "defines_body":
                            is_defines_body = True
                            break

                    if not is_defines_body:
                        continue

                    # Found an implementation!
                    impl_data = self.graph.nodes[impl_node]
                    impl_file = impl_data.get("file_path", "")

                    # Check if implementation is in a different file (out-of-line)
                    if not impl_file or impl_file == class_file:
                        continue  # Skip inline or same-file implementations

                    # Get source text from symbol object
                    symbol_obj = impl_data.get("symbol")
                    if not symbol_obj or not hasattr(symbol_obj, "source_text"):
                        continue

                    source_text = symbol_obj.source_text
                    if not source_text or not source_text.strip():
                        continue

                    rel_path = impl_data.get("rel_path", impl_file)

                    # Add to collection grouped by file
                    if rel_path not in impl_by_file:
                        impl_by_file[rel_path] = []
                    impl_by_file[rel_path].append(source_text)
                    processed_nodes.add(impl_node)

                    method_name = impl_data.get("symbol_name", "unknown")
                    self.logger.debug(f"Collected out-of-line implementation for {method_name} from {rel_path}")

        except Exception as e:
            self.logger.error(f"Error collecting C++ method implementations: {e}", exc_info=True)

        return impl_by_file

    def _augment_cpp_function_with_implementation(self, function_node: str, function_doc: Document) -> Document | None:
        """
        Augment C++ function declaration with implementation from .cpp file.

        For C++ functions with split declaration/implementation:
        - Header (.h): Function declaration
        - Implementation (.cpp): Function definition with body

        This method uses DEFINES_BODY relationships to find and concatenate the implementation.

        Args:
            function_node: Graph node for the function
            function_doc: Original function document (declaration from header)

        Returns:
            Augmented document with implementation, or None if no cross-file implementation
        """
        if not self.graph or not self.graph.has_node(function_node):
            return None

        node_data = self.graph.nodes[function_node]
        function_name = node_data.get("symbol_name", "")
        function_file = node_data.get("file_path", "")

        # Look for DEFINES_BODY relationships pointing FROM implementations TO this declaration
        # Note: DEFINES_BODY goes implementation→declaration, so we need incoming edges
        implementation_source = None
        implementation_file = None

        try:
            # Find nodes that have DEFINES_BODY edges to this function
            for source_node, _target_node, edge_data in self.graph.in_edges(function_node, data=True):
                rel_type = edge_data.get("relationship_type", "")
                if rel_type == "defines_body":
                    # Found an implementation
                    source_data = self.graph.nodes[source_node]
                    source_symbol = source_data.get("symbol", None)

                    if source_symbol and hasattr(source_symbol, "source_text"):
                        impl_file = source_data.get("file_path", "")

                        # Only augment if implementation is in a different file (cross-file)
                        if impl_file and impl_file != function_file:
                            implementation_source = source_symbol.source_text
                            implementation_file = source_data.get("rel_path", impl_file)
                            break

            # If we found a cross-file implementation, augment the document
            if implementation_source and implementation_file:
                augmented_content = function_doc.page_content
                augmented_content += f"\n\n/* Implementation from {implementation_file} */\n"
                augmented_content += implementation_source

                # Create new document with augmented content
                augmented_metadata = function_doc.metadata.copy()
                augmented_metadata["has_split_implementation"] = True
                augmented_metadata["implementation_file"] = implementation_file
                augmented_metadata["replaces_original"] = True

                return Document(page_content=augmented_content, metadata=augmented_metadata)

        except Exception as e:
            self.logger.error(f"Error augmenting function {function_name} with implementation: {e}", exc_info=True)

        return None

    def _augment_cpp_constant_with_definition(self, constant_node: str, constant_doc: Document) -> Document | None:
        """
        Augment C++ constant declaration with definition from .cpp file.

        For C++ constants with split declaration/definition:
        - Header (.h): extern const declaration
        - Implementation (.cpp): const definition with actual value

        This method searches for a node with the same constant name in a different file.

        Args:
            constant_node: Graph node for the constant
            constant_doc: Original constant document (declaration from header)

        Returns:
            Augmented document with definition, or None if no cross-file definition
        """
        if not self.graph or not self.graph.has_node(constant_node):
            return None

        node_data = self.graph.nodes[constant_node]
        constant_name = node_data.get("symbol_name", "")
        constant_file = node_data.get("file_path", "")
        full_name = node_data.get("full_name", constant_name)

        # Search for other nodes with the same constant name in different files
        definition_source = None
        definition_file = None

        self._ensure_graph_indexes()

        try:
            candidates = []

            # Prefer explicit decl->impl index if available
            if self._decl_impl_index and constant_node in self._decl_impl_index:
                candidates = list(self._decl_impl_index.get(constant_node, []))
            elif self._constant_def_index and constant_name in self._constant_def_index:
                candidates = list(self._constant_def_index.get(constant_name, []))

            for other_node in candidates:
                if other_node == constant_node:
                    continue

                other_data = self.graph.nodes.get(other_node, {})
                other_symbol_type = other_data.get("symbol_type", "")
                other_name = other_data.get("symbol_name", "")
                other_full_name = other_data.get("full_name", other_name)
                other_file = other_data.get("file_path", "")

                if (
                    other_symbol_type == "constant"
                    and (other_name == constant_name or other_full_name == full_name)
                    and other_file
                    and other_file != constant_file
                ):
                    other_symbol = other_data.get("symbol", None)
                    if other_symbol and hasattr(other_symbol, "source_text"):
                        definition_source = other_symbol.source_text
                        definition_file = other_data.get("rel_path", other_file)
                        break

            # If we found a cross-file definition, augment the document
            if definition_source and definition_file:
                augmented_content = constant_doc.page_content
                augmented_content += f"\n\n/* Definition from {definition_file} */\n"
                augmented_content += definition_source

                # Create new document with augmented content
                augmented_metadata = constant_doc.metadata.copy()
                augmented_metadata["has_split_definition"] = True
                augmented_metadata["definition_file"] = definition_file
                augmented_metadata["replaces_original"] = True

                return Document(page_content=augmented_content, metadata=augmented_metadata)

        except Exception as e:
            self.logger.error(f"Error augmenting constant {constant_name} with definition: {e}", exc_info=True)

        return None

    def _find_struct_document(self, struct_node: str) -> Document | None:
        """
        Find or create the document for a struct node.

        Args:
            struct_node: Graph node for the struct

        Returns:
            Document for the struct, or None
        """
        try:
            return self._create_document_from_graph_node(struct_node)
        except Exception as e:
            self.logger.error(f"Error finding struct document: {e}")
            return None

    def _augment_struct_with_implementations(
        self, struct_doc: Document, impl_by_file: dict[str, list[str]]
    ) -> Document:
        """
        Augment struct document with method implementations from separate files.

        Creates a new document with:
        - Original struct declaration
        - Method implementations grouped by file with clear attribution

        Args:
            struct_doc: Original struct document
            impl_by_file: Dict mapping rel_path -> list of method source texts

        Returns:
            New document with augmented content
        """
        # Build augmented content
        augmented_content = struct_doc.page_content
        augmented_content += "\n\n/* Method Implementations */\n"

        # Add implementations grouped by file (using relative paths)
        for rel_path, methods in sorted(impl_by_file.items()):
            augmented_content += f"\n// From {rel_path}\n"
            for method_source in methods:
                augmented_content += method_source.rstrip() + "\n\n"

        # Create new document with augmented content
        augmented_metadata = struct_doc.metadata.copy()
        augmented_metadata["has_split_implementations"] = True
        augmented_metadata["implementation_files"] = list(impl_by_file.keys())  # List of relative paths
        augmented_metadata["replaces_original"] = True  # Signal to replace original struct doc

        return Document(page_content=augmented_content, metadata=augmented_metadata)

    def _expand_method_comprehensively(
        self,
        method_node: str,
        processed_nodes: set[str],
        original_doc: Document = None,
        existing_symbols: set[str] = None,
    ) -> list[Document]:
        """
        Provide comprehensive ARCHITECTURAL context for a function/method.

        NOTE: Called for both 'method' and 'function' symbols, but:
        - Functions (architectural): ARE in vector store, expansion executes
        - Methods (non-architectural): NOT in vector store, expansion never executes

        Best-in-class documentation for a function should include:
        - Parent class context (if function is a class method)
        - Return type class (if function returns object type) ✨ NEW
        - Parameter type classes (if function takes object parameters) ✨ NEW
        - Functions this function calls
        - Functions that call this function
        - Overridden methods from parent classes

        Args:
            method_node: Graph node ID for the method/function
            processed_nodes: Set of already processed nodes to avoid duplicates
            original_doc: Original document containing metadata about specific method instance
            existing_symbols: Set of symbols already in retrieved documents to avoid duplicates
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(method_node):
            return context_docs

        node_data = self.graph.nodes[method_node]
        method_name = node_data.get("symbol_name", "")
        language = node_data.get("language", "")

        self.logger.debug(f"Expanding method {method_name} comprehensively")

        # C++ SPECIFIC: Augment function declaration with implementation if split across files
        if language == "cpp" and original_doc:
            try:
                augmented_doc = self._augment_cpp_function_with_implementation(method_node, original_doc)
                if augmented_doc and augmented_doc.page_content != original_doc.page_content:
                    # Add augmented version to context (will replace original in retrieval results)
                    context_docs.append(augmented_doc)
                    self.logger.debug(f"Augmented C++ function {method_name} with implementation")
            except Exception as e:
                self.logger.error(f"Error augmenting C++ function with implementation: {e}", exc_info=True)

        # 1. Find parent class (essential context) - use original doc metadata if available
        parent_docs = self._find_parent_context_for_method(method_node, processed_nodes, original_doc)
        # Filter out duplicates with retrieved documents
        parent_docs = self._filter_unique_documents(parent_docs, existing_symbols)
        context_docs.extend(parent_docs)

        # 2. NEW: Expand to return type class (e.g., getUserProfile() -> UserProfile)
        return_type_docs = self._expand_return_type(method_node, processed_nodes, existing_symbols)
        context_docs.extend(return_type_docs)

        # 3. NEW: Expand to parameter type classes (e.g., process(data: DataFrame) -> DataFrame)
        param_type_docs = self._expand_parameter_types(method_node, processed_nodes, existing_symbols)
        context_docs.extend(param_type_docs)

        # 3b. NEW: Expand to composed classes of parameter types (transitive field access)
        # e.g., function(config: AppConfig) where AppConfig has db: DatabaseConfig -> DatabaseConfig
        composed_class_docs = self._expand_composed_classes_of_parameters(
            method_node, param_type_docs, processed_nodes, existing_symbols
        )
        context_docs.extend(composed_class_docs)

        # 4. NEW: Expand to constructed/created classes (e.g., user = new User() -> User)
        created_class_docs = self._expand_created_classes(method_node, processed_nodes, existing_symbols)
        context_docs.extend(created_class_docs)

        # 5. Find methods this method calls (including constructors)
        # Limit increased to 5 to accommodate typical patterns: 2-3 helper functions + 2-3 object constructions
        called_methods = self._find_called_methods(method_node, processed_nodes, limit=5)
        # Filter out duplicates with retrieved documents
        called_methods = self._filter_unique_documents(called_methods, existing_symbols)
        logger.debug(f"Found {len(called_methods)} unique called methods")
        context_docs.extend(called_methods)

        # 6. Find methods/classes that call this method (callers)
        # Limit of 3 provides good examples of usage without bloating context
        caller_docs = self._find_method_callers(method_node, processed_nodes, limit=3)
        # Filter out duplicates with retrieved documents
        caller_docs = self._filter_unique_documents(caller_docs, existing_symbols)
        logger.debug(f"Found {len(caller_docs)} unique method callers")
        context_docs.extend(caller_docs)

        # 7. Find overridden methods in parent classes
        overridden_docs = self._find_overridden_methods(method_node, processed_nodes)
        # Filter out duplicates with retrieved documents
        overridden_docs = self._filter_unique_documents(overridden_docs, existing_symbols)
        logger.debug(f"Found {len(overridden_docs)} unique overridden methods")
        context_docs.extend(overridden_docs)

        logger.debug(
            f"Method {method_name} expanded with {len(context_docs)} additional documents "
            f"(including {len(return_type_docs)} return type, {len(param_type_docs)} parameter types)"
        )
        return context_docs

    def _expand_constructor_comprehensively(
        self, constructor_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand constructors to provide comprehensive context.

        Handles two cases:
        1. Constructor Definition (def __init__(self):) → expand to enclosing class
        2. Constructor Call/Invocation (MyClass() in method) → expand to:
           - The class being instantiated
           - The enclosing method containing the call
           - The class containing that method

        Args:
            constructor_node: Graph node ID for the constructor
            processed_nodes: Set of already processed nodes to avoid duplicates

        Returns:
            List of documents providing comprehensive constructor context
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(constructor_node):
            return context_docs

        constructor_data = self.graph.nodes[constructor_node]
        constructor_name = constructor_data.get("symbol_name", "Unknown")
        constructor_data.get("symbol_type", "").lower()

        # Determine if this is a constructor call or constructor definition
        is_constructor_call = self._is_constructor_call(constructor_node, constructor_data)

        self.logger.debug(f"Constructor node: {constructor_node}")
        self.logger.debug(f"Constructor data: {constructor_data}")
        self.logger.debug(f"Is constructor call: {is_constructor_call}")

        if is_constructor_call:
            # Handle constructor call: MyClass() inside another method
            self.logger.debug(f"Expanding constructor call {constructor_name}")
            context_docs.extend(self._expand_constructor_call(constructor_node, processed_nodes))
        else:
            # Handle constructor definition: def __init__(self):
            self.logger.debug(f"Expanding constructor definition {constructor_name}")
            context_docs.extend(self._expand_constructor_definition(constructor_node, processed_nodes))

        self.logger.debug(f"Constructor {constructor_name} expanded with {len(context_docs)} additional documents")
        return context_docs

    def _is_constructor_call(self, constructor_node: str, constructor_data: dict) -> bool:
        """
        Determine if this is a constructor call vs constructor definition.

        Based on our investigation, constructor detection should use:
        - Nodes ending with .<init> suffix are constructor definitions
        - Methods calling constructor definitions via 'calls' relationship are constructor calls
        """
        symbol_name = constructor_data.get("symbol_name", "")
        symbol_type = constructor_data.get("symbol_type", "").lower()

        # Check if this is a constructor definition (ends with .<init>)
        if symbol_name.endswith(".<init>"):
            return False  # This is a constructor definition, not a call

        # Direct check for constructor_call type
        if "constructor_call" in symbol_type:
            return True

        # Check for call-related types
        if any(keyword in symbol_type for keyword in ["call", "invocation", "instantiation"]):
            return True

        # Check if this method calls any constructor definitions (nodes ending with .<init>)
        # Look for outgoing edges to .<init> nodes
        for succ in self.graph.successors(constructor_node):
            succ_data = self.graph.nodes.get(succ, {})
            succ_name = succ_data.get("symbol_name", "")
            if succ_name.endswith(".<init>"):
                # Check edge type
                edge_data = self.graph.get_edge_data(constructor_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type in ["calls", "invokes", "instantiates"]:
                            return True

        return False

    def _expand_constant_comprehensively(
        self, constant_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand constants to provide initialization context (Strategy 3).

        Rationale:
        Constants often call functions or reference types during initialization:
        - const DEFAULT_USER = createDefaultUser() → bring createDefaultUser into context
        - const MAX_SIZE: number = calculateLimit() → bring calculateLimit into context
        - const CONFIG = { db: new Database() } → bring Database into context

        This implements "bring initialization dependencies into scope" strategy.

        Args:
            constant_node: The constant node to expand
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context

        Returns:
            List of Document objects for initialization functions and types
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(constant_node):
            return context_docs

        constant_data = self.graph.nodes.get(constant_node, {})
        constant_name = constant_data.get("symbol_name", "")

        self.logger.debug(f"Expanding constant {constant_name}")

        # 1. Find functions called during initialization (CALLS edges)
        for succ in self.graph.successors(constant_node):
            if succ in processed_nodes:
                continue

            edge_data = self.graph.get_edge_data(constant_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type in ["calls", "invokes", "references"]:
                        succ_data = self.graph.nodes.get(succ, {})
                        succ_type = succ_data.get("symbol_type", "").lower()
                        succ_name = succ_data.get("symbol_name", "")

                        # Include functions and classes (constructors)
                        if succ_type in ["function", "class"]:
                            # Skip built-ins
                            file_path = succ_data.get("file_path", "")
                            if file_path and not file_path.startswith("<") and "built-in" not in file_path.lower():
                                func_doc = self._create_document_from_graph_node(
                                    succ, expansion_reason="initializes", source_symbol=constant_name
                                )
                                if func_doc and not self._is_symbol_already_retrieved(func_doc, existing_symbols):
                                    context_docs.append(func_doc)
                                    processed_nodes.add(succ)
                                    self.logger.debug(f"Added initialization dependency: {succ_name} ({succ_type})")
                                    break

        # 2. Find type of the constant (if it has type annotation)
        # Look for REFERENCES edges to type classes
        for succ in self.graph.successors(constant_node):
            if succ in processed_nodes or len(context_docs) >= 3:
                continue

            edge_data = self.graph.get_edge_data(constant_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type == "references":
                        succ_data = self.graph.nodes.get(succ, {})
                        succ_type = succ_data.get("symbol_type", "").lower()

                        # Include type definitions
                        if succ_type in ["class", "interface", "enum", "type_alias"]:
                            # For type aliases, follow ALIAS_OF chain to resolve to concrete types
                            if succ_type == "type_alias":
                                resolved_targets = self._resolve_alias_chain(succ)
                                for target_node in resolved_targets:
                                    if target_node in processed_nodes:
                                        continue
                                    target_data = self.graph.nodes.get(target_node, {})
                                    target_file = target_data.get("file_path", "")
                                    if not target_file or target_file.startswith("<"):
                                        continue
                                    target_doc = self._create_document_from_graph_node(
                                        target_node,
                                        expansion_reason="type_of_alias_target",
                                        source_symbol=constant_name,
                                    )
                                    if target_doc and not self._is_symbol_already_retrieved(
                                        target_doc, existing_symbols
                                    ):
                                        context_docs.append(target_doc)
                                        processed_nodes.add(target_node)
                                        self.logger.debug(
                                            f"Added constant type via alias: {target_data.get('symbol_name')}"
                                        )
                                # Also include the alias itself for context
                            type_doc = self._create_document_from_graph_node(
                                succ, expansion_reason="type_of", source_symbol=constant_name
                            )
                            if type_doc and not self._is_symbol_already_retrieved(type_doc, existing_symbols):
                                context_docs.append(type_doc)
                                processed_nodes.add(succ)
                                self.logger.debug(f"Added constant type: {succ_data.get('symbol_name')}")
                                break

        return context_docs

    def _expand_constructor_call(self, call_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Expand constructor call like MyClass() inside OtherClass.__init__().

        Should expand to:
        1. MyClass (the class being instantiated)
        2. __init__ method containing the call (enclosing method)
        3. OtherClass (class containing the enclosing method)
        """
        context_docs = []

        # 1. Find the class being instantiated (target of the constructor call)
        instantiated_class_docs = self._find_instantiated_class(call_node, processed_nodes)
        context_docs.extend(instantiated_class_docs)

        # 2. Find the enclosing method that contains this constructor call
        enclosing_method_docs = self._find_parent_context(call_node, processed_nodes)
        context_docs.extend(enclosing_method_docs)

        # 3. Find the class containing the enclosing method (go up one more level)
        for method_doc in enclosing_method_docs:
            method_symbol_info = self._extract_symbol_info_from_document(method_doc)
            if method_symbol_info:
                method_name, method_type, file_path, language = method_symbol_info
                if method_type.lower() in ["method", "function", "constructor"]:
                    method_node = self._find_graph_node(method_name, file_path, language)
                    if method_node:
                        # Find the class containing this method
                        containing_class_docs = self._find_parent_context(method_node, processed_nodes)
                        context_docs.extend(containing_class_docs)
                        break  # Only need one containing class

        return context_docs

    def _expand_constructor_definition(self, constructor_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Expand constructor definition like def __init__(self): to its enclosing class.

        This is the original logic for constructor definitions.
        """
        context_docs = []

        # 1. Find the enclosing class - this is the primary context for constructor definitions
        parent_class_docs = self._find_parent_context(constructor_node, processed_nodes)
        context_docs.extend(parent_class_docs)

        # 2. Find other constructors in the same class (constructor overloading)
        if parent_class_docs:
            class_node = None
            # Find the class node from parent containers
            for doc in parent_class_docs:
                # Extract symbol info and find code_graph node
                symbol_info = self._extract_symbol_info_from_document(doc)
                if symbol_info:
                    symbol_name, symbol_type, file_path, language = symbol_info
                    if symbol_type.lower() == "class":
                        class_node = self._find_graph_node(symbol_name, file_path, language)
                        break

            if class_node:
                # Find other constructors in the same class
                other_constructors = self._find_constructors(class_node, processed_nodes)
                # Filter out the current constructor to avoid duplicates
                for ctor_doc in other_constructors:
                    ctor_symbol_info = self._extract_symbol_info_from_document(ctor_doc)
                    if ctor_symbol_info:
                        ctor_name, ctor_type, ctor_file, ctor_lang = ctor_symbol_info
                        ctor_node = self._find_graph_node(ctor_name, ctor_file, ctor_lang)
                        if ctor_node != constructor_node:
                            context_docs.append(ctor_doc)

        return context_docs

    def _find_instantiated_class(self, call_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Find the class that is being instantiated by a constructor call.

        For MyClass() call, find the MyClass class definition.
        """
        instantiated_docs = []

        call_data = self.graph.nodes.get(call_node, {})

        # Look for outgoing edges to classes that this call instantiates
        for succ in self.graph.successors(call_node):
            if succ in processed_nodes:
                continue

            edge_data = self.graph.get_edge_data(call_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type in ["instantiates", "creates", "calls"]:
                        succ_data = self.graph.nodes.get(succ, {})
                        succ_type = succ_data.get("symbol_type", "").lower()
                        if succ_type in ["class", "interface"]:
                            instantiated_doc = self._create_document_from_graph_node(
                                succ, expansion_reason="instantiated_by", source_symbol=call_data.get("symbol_name", "")
                            )
                            if instantiated_doc:
                                instantiated_docs.append(instantiated_doc)
                                processed_nodes.add(succ)

        # Alternative: look for class with same name as constructor call
        if not instantiated_docs:
            call_data = self.graph.nodes[call_node]
            class_name = call_data.get("symbol_name", "")
            if class_name:
                class_node = self._find_graph_node(
                    class_name, call_data.get("file_path", ""), call_data.get("language", "")
                )
                if class_node and class_node not in processed_nodes:
                    class_doc = self._create_document_from_graph_node(
                        class_node, expansion_reason="instantiated_by", source_symbol=class_name
                    )
                    if class_doc:
                        instantiated_docs.append(class_doc)
                        processed_nodes.add(class_node)

        return instantiated_docs

    def _expand_small_symbol_comprehensively(
        self, symbol_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand small symbols (variables, fields) to their most meaningful containing context.

        For variables: method + class context
        For fields: class context + related methods that use this field
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(symbol_node):
            return context_docs

        node_data = self.graph.nodes[symbol_node]
        symbol_name = node_data.get("symbol_name", "")
        symbol_type = node_data.get("symbol_type", "").lower()

        self.logger.debug(f"Expanding small symbol {symbol_name} ({symbol_type}) comprehensively")

        if symbol_type in ["variable", "parameter"]:
            # Variables need method and class context
            parent_context = self._find_parent_context(symbol_node, processed_nodes)
            parent_context = self._filter_unique_documents(parent_context, existing_symbols)
            context_docs.extend(parent_context)

        elif symbol_type in ["field", "property"]:
            # Fields need class context + methods that use them
            parent_context = self._find_parent_context(symbol_node, processed_nodes)
            parent_context = self._filter_unique_documents(parent_context, existing_symbols)
            context_docs.extend(parent_context)

            methods_using_field = self._find_methods_using_field(symbol_node, processed_nodes, limit=2)
            methods_using_field = self._filter_unique_documents(methods_using_field, existing_symbols)
            context_docs.extend(methods_using_field)

        self.logger.debug(f"Small symbol {symbol_name} expanded with {len(context_docs)} additional documents")
        return context_docs

    def _expand_type_definition_comprehensively(
        self, type_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand type definitions (enums, annotations) with usage context.
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(type_node):
            return context_docs

        node_data = self.graph.nodes[type_node]
        type_name = node_data.get("symbol_name", "")

        self.logger.debug(f"Expanding type definition {type_name} comprehensively")

        # Find classes/methods that use this type
        usage_docs = self._find_type_usage_context(type_node, processed_nodes, limit=2)
        usage_docs = self._filter_unique_documents(usage_docs, existing_symbols)
        context_docs.extend(usage_docs)

        return context_docs

    def _resolve_alias_chain(self, alias_node: str, max_depth: int = 5) -> list[str]:
        """
        Follow ALIAS_OF edges transitively to find the ultimate non-alias target type(s).

        Given a type_alias node, follows its outgoing ALIAS_OF edges to find the
        concrete types it resolves to. Handles chains like:
            using A = B;  // B is also an alias
            using B = MyClass;  // MyClass is a class
        Result: [node_id_of_MyClass]

        Also handles aliases to templated types where the alias points to multiple targets:
            using MyVec = vector<MyClass>;  // ALIAS_OF → MyClass (vector is built-in, skipped)

        Args:
            alias_node: Starting type_alias node ID
            max_depth: Maximum chain depth to prevent infinite loops

        Returns:
            List of resolved non-alias node IDs (classes, structs, enums, interfaces, etc.)
        """
        if not self.graph or not self.graph.has_node(alias_node):
            return []

        resolved = []
        visited = {alias_node}
        frontier = [alias_node]
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = []
            for current in frontier:
                for succ in self.graph.successors(current):
                    if succ in visited:
                        continue

                    edge_data = self.graph.get_edge_data(current, succ)
                    if not edge_data:
                        continue

                    # Check if any edge is ALIAS_OF
                    has_alias_of = False
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type == "alias_of":
                            has_alias_of = True
                            break

                    if not has_alias_of:
                        continue

                    visited.add(succ)
                    succ_data = self.graph.nodes.get(succ, {})
                    succ_type = succ_data.get("symbol_type", "").lower()

                    if succ_type == "type_alias":
                        # Continue following the chain
                        next_frontier.append(succ)
                    elif succ_type in ("class", "struct", "interface", "enum", "trait", "protocol", "function"):
                        # Found a concrete target
                        resolved.append(succ)
                    # else: skip primitives / built-ins that don't have file_path

            frontier = next_frontier
            depth += 1

        return resolved

    def _expand_type_alias_comprehensively(
        self, alias_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand type alias by following ALIAS_OF chain to bring underlying types into context.

        When a type alias like `using MyVec = vector<MyClass>` appears in search results,
        this method:
        1. Follows ALIAS_OF edges to find concrete types (e.g. MyClass)
        2. Brings those types into documentation context
        3. Also finds where this alias is used (functions taking it as parameter, etc.)

        This makes type aliases architecturally useful — instead of just showing
        "using MyVec = vector<MyClass>", the documentation also includes MyClass definition.

        Args:
            alias_node: The type_alias graph node
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context

        Returns:
            List of Document objects for the resolved underlying types and usage context
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(alias_node):
            return context_docs

        node_data = self.graph.nodes[alias_node]
        alias_name = node_data.get("symbol_name", "")

        self.logger.debug(f"Expanding type alias {alias_name} comprehensively")

        # 1. Follow ALIAS_OF chain to find concrete types
        resolved_targets = self._resolve_alias_chain(alias_node)

        for target_node in resolved_targets:
            if target_node in processed_nodes:
                continue

            target_data = self.graph.nodes.get(target_node, {})
            target_name = target_data.get("symbol_name", "")
            target_file = target_data.get("file_path", "")

            # Skip built-ins (no file path or built-in path)
            if not target_file or target_file.startswith("<") or "built-in" in target_file.lower():
                continue

            target_doc = self._create_document_from_graph_node(
                target_node,
                expansion_reason="alias_target",
                source_symbol=alias_name,
                via_context=f"via type alias {alias_name}",
            )
            if target_doc and not self._is_symbol_already_retrieved(target_doc, existing_symbols):
                context_docs.append(target_doc)
                processed_nodes.add(target_node)
                self.logger.debug(f"Added alias target: {target_name} via ALIAS_OF from {alias_name}")

        # 2. Find usage context — classes/functions that reference this alias
        usage_docs = self._find_type_usage_context(alias_node, processed_nodes, limit=2)
        usage_docs = self._filter_unique_documents(usage_docs, existing_symbols)
        context_docs.extend(usage_docs)

        return context_docs

    def _find_inheritance_context(self, class_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Find parent classes, implemented interfaces, and child classes.

        Recursively expands inheritance chains to include entire hierarchy:
        - User → TimestampedEntity → Entity (complete chain)
        - Shape → Drawable, Renderable (all interfaces)
        """
        inheritance_docs = []

        # Get file path for this class to check file-level relationships
        node_data = self.graph.nodes[class_node]
        class_file_path = node_data.get("file_path", "")
        class_language = node_data.get("language", "")
        class_name = node_data.get("symbol_name", class_node.split("::")[-1])

        # Find the file-level node for this class
        file_node = None
        if class_file_path:
            file_name = os.path.splitext(os.path.basename(class_file_path))[0]
            possible_file_nodes = [
                f"{class_language}::{file_name}::__file__",
                f"{class_language}::{file_name}::__file__".replace("::", "_"),
            ]
            for possible_node in possible_file_nodes:
                if self.graph.has_node(possible_node):
                    file_node = possible_node
                    break

        # Function to check relationships from a given node and recurse
        def check_relationships_from_node(source_node, current_depth=0, max_depth=5, source_class_name=class_name):
            """Recursively find inheritance relationships with depth limit."""
            if current_depth >= max_depth:
                return []

            local_docs = []
            # Look for inheritance and implements relationships (outgoing from node)
            for succ in self.graph.successors(source_node):
                if succ in processed_nodes:
                    continue

                edge_data = self.graph.get_edge_data(source_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type in ["inheritance", "extends", "implements", "implementation"]:
                            # Determine expansion reason based on relationship direction
                            expansion_reason = (
                                "extended_by" if rel_type in ["inheritance", "extends"] else "implemented_by"
                            )
                            parent_doc = self._create_document_from_graph_node(
                                succ, expansion_reason=expansion_reason, source_symbol=source_class_name
                            )
                            if parent_doc:
                                local_docs.append(parent_doc)
                                processed_nodes.add(succ)

                                # Recursively find parents of this parent (grandparents, etc.)
                                succ_name = self.graph.nodes[succ].get("symbol_name", succ.split("::")[-1])
                                grandparent_docs = check_relationships_from_node(
                                    succ, current_depth + 1, max_depth, succ_name
                                )
                                local_docs.extend(grandparent_docs)
            return local_docs

        # Check relationships from the class node itself (recursive)
        inheritance_docs.extend(check_relationships_from_node(class_node))

        # Also check relationships from the file node (for interface implementations)
        if file_node and file_node != class_node:
            inheritance_docs.extend(check_relationships_from_node(file_node))

        # Also look for incoming inheritance relationships (classes that inherit from this class)
        for pred in self.graph.predecessors(class_node):
            if pred in processed_nodes or len(inheritance_docs) >= 6:
                continue

            edge_data = self.graph.get_edge_data(pred, class_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type in ["inheritance", "extends"]:
                        # This class is a parent - the child extends this class
                        child_doc = self._create_document_from_graph_node(
                            pred,
                            expansion_reason="extends",  # child extends class_node
                            source_symbol=class_name,
                        )
                        if child_doc:
                            inheritance_docs.append(child_doc)
                            processed_nodes.add(pred)

        # For interfaces: also look for implementation relationships (incoming to interface)
        symbol_type = node_data.get("symbol_type", "").lower()
        if symbol_type == "interface":
            for pred in self.graph.predecessors(class_node):
                if len(inheritance_docs) >= 6:  # Limit for implementations but don't skip processed nodes
                    continue

                edge_data = self.graph.get_edge_data(pred, class_node)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type in ["implementation", "implements"]:
                            pred_node_data = self.graph.nodes[pred]
                            pred_symbol_type = pred_node_data.get("symbol_type", "").lower()

                            # For interfaces, we want implementing classes even if they're processed elsewhere
                            # Only skip if we've already added this specific node in this expansion
                            if pred in processed_nodes:
                                # Check if this implementing class is already in our inheritance_docs
                                already_added = False
                                for existing_doc in inheritance_docs:
                                    existing_symbol_info = self._extract_symbol_info_from_document(existing_doc)
                                    if existing_symbol_info:
                                        existing_name, existing_type, existing_file, existing_lang = (
                                            existing_symbol_info
                                        )
                                        existing_node = self._find_graph_node(
                                            existing_name, existing_file, existing_lang
                                        )
                                        if existing_node == pred:
                                            already_added = True
                                            break

                                if already_added:
                                    continue

                            # If this is a file-level implementation, find classes within that file
                            if pred_symbol_type in ["module", "file"]:
                                # Use DEFINES edges to find classes in this module (O(degree) instead of O(N))
                                # Week 7: Modules now create DEFINES edges to their top-level classes
                                for succ in self.graph.successors(pred):
                                    # Skip if we already have too many implementations
                                    if len(inheritance_docs) >= 6:
                                        break

                                    edge_data_succ = self.graph.get_edge_data(pred, succ)
                                    if edge_data_succ:
                                        for edge_succ in edge_data_succ.values():
                                            rel_type_succ = edge_succ.get("relationship_type", "").lower()
                                            if rel_type_succ == "defines":
                                                succ_data = self.graph.nodes.get(succ, {})
                                                succ_type = succ_data.get("symbol_type", "").lower()

                                                # Only include classes
                                                if succ_type == "class":
                                                    # Check if we already added this class
                                                    already_added = False
                                                    for existing_doc in inheritance_docs:
                                                        existing_symbol_info = self._extract_symbol_info_from_document(
                                                            existing_doc
                                                        )
                                                        if existing_symbol_info:
                                                            (
                                                                existing_name,
                                                                existing_type,
                                                                existing_file,
                                                                existing_lang,
                                                            ) = existing_symbol_info
                                                            existing_node = self._find_graph_node(
                                                                existing_name, existing_file, existing_lang
                                                            )
                                                            if existing_node == succ:
                                                                already_added = True
                                                                break

                                                    if not already_added:
                                                        class_doc = self._create_document_from_graph_node(succ)
                                                        if class_doc:
                                                            inheritance_docs.append(class_doc)
                                                            processed_nodes.add(succ)
                                                break  # Only process DEFINES edge once
                            else:
                                # Direct class implementation
                                impl_doc = self._create_document_from_graph_node(pred)
                                if impl_doc:
                                    inheritance_docs.append(impl_doc)
                                    processed_nodes.add(pred)

        return inheritance_docs[:4]  # Increased limit for interfaces with implementations

    def _find_constructors(self, class_node: str, processed_nodes: set[str]) -> list[Document]:
        """Find constructors for a class."""
        constructor_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            return constructor_docs

        node_data = self.graph.nodes[class_node]
        class_name = node_data.get("symbol_name", "")

        logger.debug(f"Looking for constructors of {class_name} using DEFINES edges")

        # Use DEFINES edges to find constructors (O(degree) instead of O(N))
        for constructor_node in self.graph.successors(class_node):
            if constructor_node in processed_nodes:
                continue

            # Check if this is a DEFINES edge
            edge_data = self.graph.get_edge_data(class_node, constructor_node)
            if not edge_data:
                continue

            # Check all edges between these nodes
            for edge in edge_data.values():
                rel_type = edge.get("relationship_type", "").lower()
                if rel_type != "defines":
                    continue

                # Check if it's a constructor
                constructor_data = self.graph.nodes[constructor_node]
                symbol_type = constructor_data.get("symbol_type", "").lower()

                if symbol_type == "constructor":
                    constructor_doc = self._create_document_from_graph_node(
                        constructor_node, expansion_reason="constructor_of", source_symbol=class_name
                    )
                    if constructor_doc:
                        constructor_docs.append(constructor_doc)
                        processed_nodes.add(constructor_node)
                        logger.debug(
                            f"Added constructor via DEFINES: {constructor_data.get('symbol_name', constructor_node)}"
                        )
                    break

        logger.debug(f"Found {len(constructor_docs)} constructors using DEFINES edges")
        return constructor_docs[:2]  # Limit to 2 constructors

    def _find_key_methods(self, class_node: str, processed_nodes: set[str], limit: int = 3) -> list[Document]:
        """Find key public methods of a class using DEFINES edges."""
        method_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            return method_docs

        node_data = self.graph.nodes[class_node]
        class_name = node_data.get("symbol_name", "")

        logger.debug(f"Looking for methods of {class_name} using DEFINES edges")

        # Use DEFINES edges to find methods (O(degree) instead of O(N))
        for method_node in self.graph.successors(class_node):
            if method_node in processed_nodes or len(method_docs) >= limit:
                continue

            # Check if this is a DEFINES edge
            edge_data = self.graph.get_edge_data(class_node, method_node)
            if not edge_data:
                continue

            # Check all edges between these nodes (can be multiple)
            for edge in edge_data.values():
                rel_type = edge.get("relationship_type", "").lower()
                if rel_type != "defines":
                    continue

                # Check if it's a method (not constructor, field, etc.)
                method_data = self.graph.nodes[method_node]
                symbol_type = method_data.get("symbol_type", "").lower()

                if symbol_type == "method":
                    method_doc = self._create_document_from_graph_node(
                        method_node, expansion_reason="method_of", source_symbol=class_name
                    )
                    if method_doc:
                        method_docs.append(method_doc)
                        processed_nodes.add(method_node)
                        logger.debug(f"Added method via DEFINES: {method_data.get('symbol_name', method_node)}")
                    break

        logger.debug(f"Found {len(method_docs)} methods using DEFINES edges")
        return method_docs

    def _find_important_fields(self, class_node: str, processed_nodes: set[str], limit: int = 2) -> list[Document]:
        """Find important fields/properties of a class using DEFINES edges."""
        field_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            return field_docs

        node_data = self.graph.nodes[class_node]
        class_name = node_data.get("symbol_name", "")

        logger.debug(f"Looking for fields of {class_name} using DEFINES edges")

        # Use DEFINES edges to find fields (O(degree) instead of O(N))
        for field_node in self.graph.successors(class_node):
            if field_node in processed_nodes or len(field_docs) >= limit:
                continue

            # Check if this is a DEFINES edge
            edge_data = self.graph.get_edge_data(class_node, field_node)
            if not edge_data:
                continue

            # Check all edges between these nodes
            for edge in edge_data.values():
                rel_type = edge.get("relationship_type", "").lower()
                if rel_type != "defines":
                    continue

                # Check if it's a field
                field_data = self.graph.nodes[field_node]
                symbol_type = field_data.get("symbol_type", "").lower()

                if symbol_type in ["field", "property"]:
                    field_doc = self._create_document_from_graph_node(
                        field_node, expansion_reason="field_of", source_symbol=class_name
                    )
                    if field_doc:
                        field_docs.append(field_doc)
                        processed_nodes.add(field_node)
                        logger.debug(f"Added field via DEFINES: {field_data.get('symbol_name', field_node)}")
                    break

        logger.debug(f"Found {len(field_docs)} fields using DEFINES edges")
        return field_docs

    def _expand_class_method_parameter_types(
        self, class_node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand to enums/classes used as parameter types in the class's methods.

        For example:
        - PermissionChecker class → hasPermission(role: Role, permission: Permission) → Role enum, Permission enum
        - UserValidator class → validate(user: User) → User interface

        This allows classes to transitionally connect to type definitions used in their method signatures.

        Handles language differences in REFERENCES edge attachment:
        - TypeScript/JavaScript: REFERENCES edges at class level
        - Python/C++: REFERENCES edges at method level

        Args:
            class_node: Graph node for the class
            processed_nodes: Already processed nodes
            existing_symbols: Already retrieved symbols

        Returns:
            List of parameter type documents (enums, interfaces, classes)
        """
        param_type_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            return param_type_docs

        class_data = self.graph.nodes.get(class_node, {})
        class_name = class_data.get("symbol_name", "")

        self.logger.debug(f"Finding parameter types for {class_name} using DEFINES edges")

        # Use DEFINES edges to find all methods of the class (O(degree) instead of O(N))
        nodes_to_check = [class_node]  # Check class node for class-level REFERENCES

        for method_node in self.graph.successors(class_node):
            # Check if this is a DEFINES edge to a method
            edge_data = self.graph.get_edge_data(class_node, method_node)
            if not edge_data:
                continue

            for edge in edge_data.values():
                rel_type = edge.get("relationship_type", "").lower()
                if rel_type == "defines":
                    # Check if it's a method or constructor (they have parameter types)
                    method_data = self.graph.nodes[method_node]
                    symbol_type = method_data.get("symbol_type", "").lower()
                    if symbol_type in ["method", "constructor", "function"]:
                        nodes_to_check.append(method_node)
                    break

        self.logger.debug(f"Checking {len(nodes_to_check)} nodes for parameter types (class + methods)")

        # Follow REFERENCES edges from all nodes (class + methods) to parameter types
        for check_node in nodes_to_check:
            for _, target_node, edge_data in self.graph.out_edges(check_node, data=True):
                rel_type = edge_data.get("relationship_type", "").lower()

                # Only follow references relationships
                if rel_type != "references":
                    continue

                # Skip if already processed
                if target_node in processed_nodes:
                    continue

                # Check if target is an enum, interface, or class (architectural type)
                target_data = self.graph.nodes.get(target_node, {})
                target_type = target_data.get("symbol_type", "").lower()

                # Only include enums, interfaces, and classes (not parameters, methods, etc.)
                if target_type not in ["enum", "interface", "class"]:
                    continue

                # Create document for the parameter type
                type_doc = self._create_document_from_graph_node(target_node)
                if type_doc:
                    param_type_docs.append(type_doc)
                    processed_nodes.add(target_node)
                    self.logger.debug(f"Added parameter type {target_data.get('symbol_name')} from class {class_name}")

        return param_type_docs

    def _find_composition_context(
        self, class_node: str, processed_nodes: set[str], limit: int = 2, depth: int = 0, max_depth: int = 2
    ) -> list[Document]:
        """
        Find classes that this class uses (composition relationships).

        Recursively expands composition to find transitive dependencies:
        - UserService.db → Database (depth 0)
        - Database.logger → Logger (depth 1)

        After parser standardization (Nov 2025), COMPOSITION edges are from FIELD nodes,
        not CLASS nodes. So we need to:
        1. Find all fields DEFINED by this class (via DEFINES edges)
        2. Follow COMPOSITION edges from those fields to find composed types

        Args:
            class_node: Graph node ID for the class
            processed_nodes: Set of already processed nodes
            limit: Maximum number of compositions per level
            depth: Current recursion depth
            max_depth: Maximum recursion depth (default 2 for transitive)

        Returns:
            List of composed class documents
        """
        composition_docs = []

        # Get class name for expansion source
        class_node_data = self.graph.nodes.get(class_node, {})
        class_name = class_node_data.get("symbol_name", class_node.split("::")[-1])

        # Stop if we've reached max depth
        if depth >= max_depth:
            return composition_docs

        # Strategy 1: Find COMPOSITION edges from class's FIELDS (standardized pattern)
        # Class --DEFINES--> Field --COMPOSITION--> Type
        for succ in self.graph.successors(class_node):
            if len(composition_docs) >= limit:
                break

            # Check if this is a DEFINES edge to a field
            edge_data = self.graph.get_edge_data(class_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    if edge.get("relationship_type", "").lower() == "defines":
                        # This is a field/method defined by the class
                        # Get field name for "via" context
                        field_node_data = self.graph.nodes.get(succ, {})
                        field_name = field_node_data.get("symbol_name", "")

                        # Now check if this field has COMPOSITION edges
                        for field_succ in self.graph.successors(succ):
                            if field_succ in processed_nodes or len(composition_docs) >= limit:
                                continue

                            field_edge_data = self.graph.get_edge_data(succ, field_succ)
                            if field_edge_data:
                                for field_edge in field_edge_data.values():
                                    rel_type = field_edge.get("relationship_type", "").lower()
                                    if rel_type in ["composition", "aggregation"]:
                                        # Extract "via" context from edge annotations
                                        annotations = field_edge.get("annotations", {})
                                        via_field = annotations.get("field_name", field_name)
                                        via_context = f"via {via_field} field" if via_field else None

                                        comp_doc = self._create_document_from_graph_node(
                                            field_succ,
                                            expansion_reason="composed_by",
                                            source_symbol=class_name,
                                            via_context=via_context,
                                        )
                                        comp_type = (
                                            (
                                                comp_doc.metadata.get("symbol_type", "")
                                                or comp_doc.metadata.get("type", "")
                                            ).lower()
                                            if comp_doc
                                            else ""
                                        )
                                        if comp_doc and comp_type in ["class", "interface", "struct"]:
                                            composition_docs.append(comp_doc)
                                            processed_nodes.add(field_succ)

                                            # Recursively find compositions of this class
                                            if depth + 1 < max_depth:
                                                transitive_comps = self._find_composition_context(
                                                    field_succ,
                                                    processed_nodes,
                                                    limit=limit,
                                                    depth=depth + 1,
                                                    max_depth=max_depth,
                                                )
                                                composition_docs.extend(transitive_comps)
                                        break  # Only need one composition per field

        # Strategy 2: Fallback - check direct edges from class (legacy/some parsers)
        # This handles cases where COMPOSITION is still attached at class level
        for succ in self.graph.successors(class_node):
            if succ in processed_nodes or len(composition_docs) >= limit:
                continue

            edge_data = self.graph.get_edge_data(class_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type in ["uses", "composition", "aggregation", "creates"]:
                        # Extract "via" context from edge if available
                        annotations = edge.get("annotations", {})
                        source_context = edge.get("source_context", "")
                        via_context = None
                        if annotations.get("field_name"):
                            via_context = f"via {annotations['field_name']} field"
                        elif source_context and "." in source_context:
                            # Extract field name from source_context like "UserService.repo"
                            via_context = f"via {source_context.split('.')[-1]} field"

                        comp_doc = self._create_document_from_graph_node(
                            succ,
                            expansion_reason="composed_by" if rel_type in ["composition", "aggregation"] else "used_by",
                            source_symbol=class_name,
                            via_context=via_context,
                        )
                        comp_type = (
                            (comp_doc.metadata.get("symbol_type", "") or comp_doc.metadata.get("type", "")).lower()
                            if comp_doc
                            else ""
                        )
                        if comp_doc and comp_type in ["class", "interface", "struct"]:
                            composition_docs.append(comp_doc)
                            processed_nodes.add(succ)

                            if depth + 1 < max_depth:
                                transitive_comps = self._find_composition_context(
                                    succ, processed_nodes, limit=limit, depth=depth + 1, max_depth=max_depth
                                )
                                composition_docs.extend(transitive_comps)

        return composition_docs

    def _find_functions_using_class(self, class_node: str, processed_nodes: set[str], limit: int = 5) -> list[Document]:
        """
        Find functions/methods that use this class (via parameters, return types, etc.).

        This looks at INCOMING relationships (predecessors) to find functions that reference the class.
        Important for finding factory functions, validators, processors, etc.
        """
        function_docs = []

        # Get class name for expansion source
        class_node_data = self.graph.nodes.get(class_node, {})
        class_name = class_node_data.get("symbol_name", class_node.split("::")[-1])

        # Look for functions that reference this class (INCOMING edges)
        for pred in self.graph.predecessors(class_node):
            if pred in processed_nodes or len(function_docs) >= limit:
                continue

            edge_data = self.graph.get_edge_data(pred, class_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    # Functions reference classes via parameters, returns, or creates
                    if rel_type in ["references", "creates", "uses"]:
                        # Determine "via" context based on relationship type
                        pred_data = self.graph.nodes.get(pred, {})
                        pred_name = pred_data.get("symbol_name", "")
                        via_context = f"via {pred_name}()" if rel_type == "creates" else None

                        func_doc = self._create_document_from_graph_node(
                            pred,
                            expansion_reason=rel_type,  # references, creates, or uses
                            source_symbol=class_name,
                            via_context=via_context,
                        )
                        func_type = (
                            (func_doc.metadata.get("symbol_type", "") or func_doc.metadata.get("type", "")).lower()
                            if func_doc
                            else ""
                        )
                        # Only include architectural symbols (functions, not methods)
                        # Methods are inside classes already, we want free-form functions
                        if func_doc and func_type in ["function"]:
                            function_docs.append(func_doc)
                            processed_nodes.add(pred)
                            break  # Only add once per predecessor

        return function_docs

    def _find_classes_using_class(self, class_node: str, processed_nodes: set[str], limit: int = 5) -> list[Document]:
        """
        Find classes that use this class (via composition, aggregation, or method usage).

        This looks at INCOMING relationships (predecessors) to find classes that reference this class.
        Important for finding:
        - Classes with fields of this type (aggregation/composition)
        - Classes whose methods use this type (transitive relationship)

        Args:
            class_node: The class node to find usages of
            processed_nodes: Set of already processed nodes
            limit: Maximum number of classes to return

        Returns:
            List of Document objects representing classes that use this class
        """
        class_docs = []

        # Get class name for expansion source
        class_node_data = self.graph.nodes.get(class_node, {})
        class_name = class_node_data.get("symbol_name", class_node.split("::")[-1])

        # Look for classes that reference this class (INCOMING edges)
        for pred in self.graph.predecessors(class_node):
            if pred in processed_nodes or len(class_docs) >= limit:
                continue

            edge_data = self.graph.get_edge_data(pred, class_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    # Classes relate to other classes via aggregation, composition, or references
                    if rel_type in ["aggregation", "composition", "references", "uses"]:
                        # Extract "via" context from edge annotations if available
                        annotations = edge.get("annotations", {})
                        source_context = edge.get("source_context", "")
                        via_context = None
                        if annotations.get("field_name"):
                            via_context = f"via {annotations['field_name']} field"
                        elif source_context and "." in source_context:
                            via_context = f"via {source_context.split('.')[-1]}"

                        class_doc = self._create_document_from_graph_node(
                            pred,
                            expansion_reason=rel_type,  # aggregation, composition, references, uses
                            source_symbol=class_name,
                            via_context=via_context,
                        )
                        class_type = (
                            (class_doc.metadata.get("symbol_type", "") or class_doc.metadata.get("type", "")).lower()
                            if class_doc
                            else ""
                        )
                        # Only include architectural class symbols
                        if class_doc and class_type in ["class", "struct"]:
                            class_docs.append(class_doc)
                            processed_nodes.add(pred)
                            break  # Only add once per predecessor

        return class_docs

    def _find_called_free_functions(
        self, class_node: str, processed_nodes: set[str], existing_symbols: set[str] = None, limit: int = 5
    ) -> list[Document]:
        """
        Find free functions that this class calls (Strategy 2: Transitive call site expansion).

        Rationale:
        When a class method calls a free function, that function provides important context
        for understanding the class's behavior. For example:
        - UserService.displayUser() calls formatUserData() → formatUserData is relevant
        - DataProcessor.validate() calls validateSchema() → validateSchema is relevant

        This implements "bring called functions into scope at call site" strategy.

        Note: CALLS edges may be attached at class level (TypeScript/JavaScript) or method level (Python/Java).
        We check both the class node and its method nodes.

        Args:
            class_node: The class node to find called functions from
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context (for deduplication)
            limit: Maximum number of functions to return

        Returns:
            List of Document objects representing called free functions
        """
        function_docs = []

        if not self.graph or not self.graph.has_node(class_node):
            return function_docs

        class_data = self.graph.nodes.get(class_node, {})
        class_name = class_data.get("symbol_name", "")

        # Collect nodes to check: class node + its method nodes
        nodes_to_check = [class_node]
        node_names = {class_node: class_name}  # Map node_id to name for "via" context

        # Use DEFINES edges to find methods (O(degree) instead of O(N) scan)
        # Week 7: All parsers now create Class → DEFINES → Method edges
        for succ in self.graph.successors(class_node):
            if succ in processed_nodes:
                continue

            edge_data = self.graph.get_edge_data(class_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get("relationship_type", "").lower()
                    if rel_type == "defines":
                        succ_data = self.graph.nodes.get(succ, {})
                        succ_type = succ_data.get("symbol_type", "")
                        succ_name = succ_data.get("symbol_name", "")

                        # Only include methods (not fields or other members)
                        if succ_type in ["method", "constructor"]:
                            nodes_to_check.append(succ)
                            node_names[succ] = succ_name
                            self.logger.debug(f"Added method via DEFINES edge: {succ}")
                            break  # Only process edge once

        self.logger.debug(f"Checking CALLS edges from {len(nodes_to_check)} nodes (class + methods) for {class_name}")

        # Look for CALLS edges from class and its methods
        for check_node in nodes_to_check:
            for succ in self.graph.successors(check_node):
                if succ in processed_nodes or len(function_docs) >= limit:
                    continue

                edge_data = self.graph.get_edge_data(check_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type in ["calls", "invokes"]:
                            succ_data = self.graph.nodes.get(succ, {})
                            succ_type = succ_data.get("symbol_type", "").lower()
                            succ_name = succ_data.get("symbol_name", "")

                            # Only include free functions (architectural symbols)
                            if succ_type == "function":
                                # Skip built-in functions
                                file_path = succ_data.get("file_path", "")
                                if not file_path or file_path.startswith("<") or "built-in" in file_path.lower():
                                    continue

                                # Skip known built-ins by name
                                builtins = {
                                    "print",
                                    "len",
                                    "range",
                                    "str",
                                    "int",
                                    "float",
                                    "list",
                                    "dict",
                                    "set",
                                    "tuple",
                                    "bool",
                                    "type",
                                    "isinstance",
                                    "hasattr",
                                    "getattr",
                                    "setattr",
                                    "open",
                                    "input",
                                    "map",
                                    "filter",
                                    "zip",
                                    "enumerate",
                                    "sorted",
                                    "sum",
                                    "min",
                                    "max",
                                    "abs",
                                    "round",
                                    "pow",
                                    "all",
                                    "any",
                                    "console.log",
                                    "console.error",
                                    "console.warn",
                                    "alert",
                                    "log",
                                }
                                if succ_name in builtins:
                                    continue

                                # Build "via" context from the calling method
                                calling_method = node_names.get(check_node, "")
                                via_context = (
                                    f"via {calling_method}()"
                                    if calling_method and calling_method != class_name
                                    else None
                                )

                                func_doc = self._create_document_from_graph_node(
                                    succ,
                                    expansion_reason="called_by",
                                    source_symbol=class_name,
                                    via_context=via_context,
                                )
                                if func_doc and not self._is_symbol_already_retrieved(func_doc, existing_symbols):
                                    function_docs.append(func_doc)
                                    processed_nodes.add(succ)
                                    self.logger.debug(f"Added called function: {succ_name}")
                                    break  # Only add once per function

        return function_docs

    def _find_called_methods(self, method_node: str, processed_nodes: set[str], limit: int = 2) -> list[Document]:
        """
        Find ARCHITECTURAL functions/methods that this method calls.

        Only includes:
        - Architectural symbols (functions in vector store)
        - NOT methods whose parent class is already in processed_nodes

        For Java/similar languages, also checks parent class edges since relationships
        may be attached at class scope rather than method scope.
        """
        called_docs = []

        # Collect nodes to check for edges (method + parent class/module for Java/JavaScript)
        nodes_to_check = [method_node]

        # For Java/JavaScript methods, also check parent class/module edges
        # (relationships may be attached at class/module scope rather than method scope)
        method_data = self.graph.nodes.get(method_node, {})
        if method_data.get("symbol_type") == "method":
            # Use DEFINES edges to find parent (O(degree) instead of O(N))
            # Week 7: All containers (class/module/namespace) now create DEFINES edges
            for pred in self.graph.predecessors(method_node):
                edge_data = self.graph.get_edge_data(pred, method_node)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type == "defines":
                            # Found parent via DEFINES edge
                            pred_data = self.graph.nodes.get(pred, {})
                            pred_type = pred_data.get("symbol_type", "")
                            if pred_type in ["class", "module", "namespace"]:
                                nodes_to_check.append(pred)
                                break

        # Look for 'calls' relationships from method and optionally parent class/module
        for check_node in nodes_to_check:
            for succ in self.graph.successors(check_node):
                if succ in processed_nodes or len(called_docs) >= limit:
                    continue

                edge_data = self.graph.get_edge_data(check_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get("relationship_type", "").lower()
                        if rel_type in ["calls", "invokes"]:
                            # Get called symbol info
                            succ_data = self.graph.nodes.get(succ, {})
                            succ_type = succ_data.get("symbol_type", "").lower()

                            # For constructor calls (e.g., User.<init>), expand to parent class
                            if succ_type == "method" and "<init>" in succ_data.get("symbol_name", ""):
                                # Use DEFINES edges to find parent class (O(degree) instead of O(N))
                                # Constructor has incoming DEFINES edge from its class
                                for pred in self.graph.predecessors(succ):
                                    if pred in processed_nodes:
                                        continue

                                    edge_data_pred = self.graph.get_edge_data(pred, succ)
                                    if edge_data_pred:
                                        for edge_pred in edge_data_pred.values():
                                            rel_type_pred = edge_pred.get("relationship_type", "").lower()
                                            if rel_type_pred == "defines":
                                                # Found parent class via DEFINES edge
                                                pred_data = self.graph.nodes.get(pred, {})
                                                if pred_data.get("symbol_type") == "class":
                                                    class_doc = self._create_document_from_graph_node(pred)
                                                    if class_doc:
                                                        called_docs.append(class_doc)
                                                        processed_nodes.add(pred)
                                                    break
                                continue  # Skip the constructor itself

                            # Skip methods - we only include functions (architectural symbols)
                            # Methods are not architectural; their parent class should be in context instead
                            if succ_type == "method":
                                # Check if parent class is already in context using DEFINES edges (O(degree))
                                parent_in_context = False
                                for pred in self.graph.predecessors(succ):
                                    edge_data_pred = self.graph.get_edge_data(pred, succ)
                                    if edge_data_pred:
                                        for edge_pred in edge_data_pred.values():
                                            if edge_pred.get("relationship_type", "").lower() == "defines":
                                                pred_data = self.graph.nodes.get(pred, {})
                                                if pred_data.get("symbol_type") == "class":
                                                    if pred in processed_nodes:
                                                        parent_in_context = True
                                                    break
                                        if parent_in_context:
                                            break
                                # Skip methods - don't add them, we already have their class or will get it
                                continue

                            # Only include architectural symbols (functions)
                            if succ_type == "function":
                                called_doc = self._create_document_from_graph_node(succ)
                                if called_doc:
                                    # Skip built-in functions (they don't have meaningful file paths)
                                    file_path = called_doc.metadata.get("file_path", "")
                                    # Built-ins typically have empty paths or special markers
                                    if not file_path or file_path.startswith("<") or "built-in" in file_path.lower():
                                        continue

                                    # Skip known built-in function names (Python, JavaScript, etc.)
                                    func_name = called_doc.metadata.get("symbol_name", "")
                                    builtins = {
                                        "print",
                                        "len",
                                        "range",
                                        "str",
                                        "int",
                                        "float",
                                        "list",
                                        "dict",
                                        "set",
                                        "tuple",
                                        "bool",
                                        "type",
                                        "isinstance",
                                        "hasattr",
                                        "getattr",
                                        "setattr",
                                        "open",
                                        "input",
                                        "map",
                                        "filter",
                                        "zip",
                                        "enumerate",
                                        "sorted",
                                        "sum",
                                        "min",
                                        "max",
                                        "abs",
                                        "round",
                                        "pow",
                                        "all",
                                        "any",
                                        "console.log",
                                        "console.error",
                                        "console.warn",
                                        "alert",
                                    }
                                    if func_name in builtins:
                                        continue

                                    called_docs.append(called_doc)
                                    processed_nodes.add(succ)

        return called_docs

    def _find_method_callers(self, method_node: str, processed_nodes: set[str], limit: int = 2) -> list[Document]:
        """Find methods and classes that call this method."""
        caller_docs = []

        if not self.graph or not self.graph.has_node(method_node):
            return caller_docs

        node_data = self.graph.nodes[method_node]
        method_name = node_data.get("symbol_name", "")
        node_data.get("file_path", "")

        # Get the class that owns this method
        method_parent = node_data.get("parent_symbol", "")
        method_class_name = method_parent.split(".")[-1] if method_parent else ""

        logger.debug(f"Looking for callers of method {method_name} in class {method_class_name}")

        # Strategy 1: Look for direct calls to this method
        for source, _target, edge_data in self.graph.in_edges(method_node, data=True):
            if len(caller_docs) >= limit:
                break

            relationship_type = edge_data.get("relationship_type", "")

            if relationship_type == "calls":
                source_node_data = self.graph.nodes[source]
                source_symbol_name = source_node_data.get("symbol_name", "")
                source_symbol_type = source_node_data.get("symbol_type", "")

                logger.debug(f"Found direct caller: {source_symbol_name} ({source_symbol_type}) calls {method_name}")

                if source not in processed_nodes:
                    caller_doc = self._create_document_from_graph_node(source)
                    if caller_doc:
                        caller_docs.append(caller_doc)
                        processed_nodes.add(source)

        # Strategy 2: Heuristic scan for calls through object references can be expensive on large graphs.
        # Disabled by default to avoid O(E^2) scans; enable explicitly if needed.
        if os.getenv("WIKIS_ENABLE_HEURISTIC_CALLER_SCAN") != "1":
            logger.debug("Skipping heuristic caller scan; set WIKIS_ENABLE_HEURISTIC_CALLER_SCAN=1 to enable")
            logger.debug(f"Found {len(caller_docs)} caller documents for method {method_name}")
            return caller_docs

        # Strategy 2: Look for method calls through object references
        # Pattern: method --[calls]--> object_var, where object_var is of type that has our method
        for source, target, edge_data in self.graph.edges(data=True):
            if len(caller_docs) >= limit:
                break

            relationship_type = edge_data.get("relationship_type", "")

            if relationship_type == "calls":
                source_node_data = self.graph.nodes[source]
                target_node_data = self.graph.nodes[target]

                source_symbol_name = source_node_data.get("symbol_name", "")
                source_symbol_type = source_node_data.get("symbol_type", "")
                target_symbol_name = target_node_data.get("symbol_name", "")
                target_symbol_type = target_node_data.get("symbol_type", "")

                # Check if source method calls a variable/parameter that could be our class type
                if (
                    source_symbol_type == "method"
                    and target_symbol_type in ["variable", "parameter"]
                    and source not in processed_nodes
                ):
                    # Look for patterns where the variable name suggests it's our class type
                    # or where there are relationships showing object instantiation
                    is_potential_caller = False

                    # Pattern 1: Variable named like our class (e.g., 'app' for 'Application')
                    if (
                        target_symbol_name.lower() in method_class_name.lower()
                        or method_class_name.lower() in target_symbol_name.lower()
                    ):
                        is_potential_caller = True
                        logger.debug(
                            f"Found potential caller by name pattern: {source_symbol_name} calls {target_symbol_name} (matches {method_class_name})"
                        )

                    # Pattern 2: Look for instantiation relationships
                    # Check incoming edges to the target variable instead of scanning all edges
                    for var_source, _var_target, var_edge_data in self.graph.in_edges(target, data=True):
                        if len(caller_docs) >= limit:
                            break
                        var_rel_type = var_edge_data.get("relationship_type", "")
                        if var_rel_type in ["calls", "references"]:
                            var_source_data = self.graph.nodes.get(var_source, {})
                            var_source_name = var_source_data.get("symbol_name", "")
                            if var_source_name == method_class_name:
                                is_potential_caller = True
                                logger.debug(
                                    f"Found caller by instantiation: {source_symbol_name} calls {target_symbol_name} of type {var_source_name}"
                                )
                                break

                    if is_potential_caller:
                        caller_doc = self._create_document_from_graph_node(source)
                        if caller_doc:
                            caller_docs.append(caller_doc)
                            processed_nodes.add(source)
                            logger.debug(f"Added caller: {source_symbol_name}")

                            # Also add the class that contains this calling method
                            parent_class_docs = self._find_parent_context(source, processed_nodes)
                            caller_docs.extend(parent_class_docs)

        logger.debug(f"Found {len(caller_docs)} caller documents for method {method_name}")
        return caller_docs

    def _find_overridden_methods(self, method_node: str, processed_nodes: set[str]) -> list[Document]:
        """
        Find methods in parent classes that this method overrides.

        INTENTIONALLY EMPTY - Methods are not architectural symbols.

        When a class is retrieved and expanded:
        1. The class source already contains its method definitions
        2. _find_inheritance_context brings parent classes into context
        3. Parent class sources already contain their method definitions

        Therefore, there's no need to separately track override relationships.
        Both the overriding method and overridden method are already included
        in their respective class sources.

        This applies to:
        - Python: class Child(Parent): def method(self): ...
        - Java: @Override public void method() { ... }
        - C++: virtual/override keywords
        - TypeScript: method implementations
        """
        return []

    def _find_type_usage_context(self, type_node: str, processed_nodes: set[str], limit: int = 2) -> list[Document]:
        """Find architectural symbols (classes, functions) that use this type definition.

        Prefers architectural symbols (classes, functions, structs) over transitive
        symbols (methods, fields, constructors) to avoid wasting expansion slots.
        When a method references this type, resolves upward to its containing class.
        """
        usage_docs = []

        if not self.graph or not self.graph.has_node(type_node):
            return usage_docs

        type_data = self.graph.nodes[type_node]
        type_data.get("symbol_name", "")
        symbol_type = type_data.get("symbol_type", "").lower()

        # Determine valid relationship types
        if symbol_type == "interface":
            valid_rels = {"references", "uses", "field_type", "implementation", "implements"}
        else:
            valid_rels = {"references", "uses", "field_type"}

        # Collect candidates: (source_node, is_architectural)
        # Architectural symbols fill slots first; non-architectural are resolved to parent
        architectural_candidates = []
        transitive_candidates = []

        for source, _target, edge_data in self.graph.in_edges(type_node, data=True):
            relationship_type = edge_data.get("relationship_type", "")
            if relationship_type not in valid_rels:
                continue
            if source in processed_nodes:
                continue

            source_data = self.graph.nodes.get(source, {})
            source_type = source_data.get("symbol_type", "").lower()

            if source_type in self.ARCHITECTURAL_SYMBOL_TYPES:
                architectural_candidates.append(source)
            else:
                # Transitive symbol (method, field, constructor, etc.)
                # Try to resolve to containing class/struct
                transitive_candidates.append(source)

        # First: add architectural sources directly
        for source in architectural_candidates:
            if len(usage_docs) >= limit:
                break
            if source in processed_nodes:
                continue
            usage_doc = self._create_document_from_graph_node(source)
            if usage_doc:
                usage_docs.append(usage_doc)
                processed_nodes.add(source)

        # Second: resolve transitive sources to their containing architectural parent
        seen_parents = set()
        for source in transitive_candidates:
            if len(usage_docs) >= limit:
                break
            # Walk up CONTAINS/DEFINES edges to find containing class/struct
            parent_node = self._resolve_to_architectural_parent(source)
            if parent_node and parent_node not in processed_nodes and parent_node not in seen_parents:
                seen_parents.add(parent_node)
                usage_doc = self._create_document_from_graph_node(parent_node)
                if usage_doc:
                    usage_docs.append(usage_doc)
                    processed_nodes.add(parent_node)

        return usage_docs

    def _resolve_to_architectural_parent(self, node: str, max_depth: int = 5) -> str | None:
        """Walk up CONTAINS/DEFINES edges to find the nearest architectural parent.

        For a method node, returns the containing class/struct.
        For an already-architectural node, returns itself.
        Returns None if no architectural parent found.
        """
        if not self.graph or not self.graph.has_node(node):
            return None

        node_data = self.graph.nodes.get(node, {})
        if node_data.get("symbol_type", "").lower() in self.ARCHITECTURAL_SYMBOL_TYPES:
            return node

        visited = {node}
        current = node
        for _ in range(max_depth):
            # Look for incoming CONTAINS or DEFINES edges (parent → child)
            parent_found = None
            for pred in self.graph.predecessors(current):
                if pred in visited:
                    continue
                edge_data = self.graph.get_edge_data(pred, current)
                if edge_data:
                    for edge in edge_data.values():
                        rel = edge.get("relationship_type", "").lower()
                        if rel in ("contains", "defines", "has_method", "has_member"):
                            parent_found = pred
                            break
                if parent_found:
                    break

            if not parent_found:
                return None

            visited.add(parent_found)
            parent_data = self.graph.nodes.get(parent_found, {})
            if parent_data.get("symbol_type", "").lower() in self.ARCHITECTURAL_SYMBOL_TYPES:
                return parent_found
            current = parent_found

        return None

    def _find_methods_using_field(self, field_node: str, processed_nodes: set[str], limit: int = 2) -> list[Document]:
        """Find methods that access or modify this field."""
        method_docs = []

        if not self.graph or not self.graph.has_node(field_node):
            return method_docs

        field_data = self.graph.nodes[field_node]
        field_data.get("symbol_name", "")

        # Look for incoming relationships where methods reference this field
        for source, _target, edge_data in self.graph.in_edges(field_node, data=True):
            if len(method_docs) >= limit:
                break

            relationship_type = edge_data.get("relationship_type", "")

            if relationship_type in ["references", "uses", "accesses", "modifies"]:
                source_data = self.graph.nodes.get(source, {})
                source_type = source_data.get("symbol_type", "").lower()

                if source_type == "method" and source not in processed_nodes:
                    method_doc = self._create_document_from_graph_node(source)
                    if method_doc:
                        method_docs.append(method_doc)
                        processed_nodes.add(source)

        return method_docs

    def _expand_return_type(
        self, node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand to return type class using symbol metadata.

        Extracts return_type from symbol metadata (e.g., 'UserProfile', 'List[User]'),
        cleans generic/template syntax, and finds the type class in graph.

        For functions: getUserProfile() returns UserProfile → bring UserProfile class
        For methods: Never executed (methods not in vector store)

        Args:
            node: Function node ID
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context (for deduplication)

        Returns:
            List of return type class documents
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(node):
            return context_docs

        node_data = self.graph.nodes.get(node, {})
        symbol = node_data.get("symbol")

        if not symbol or not hasattr(symbol, "return_type"):
            return context_docs

        return_type = symbol.return_type

        # Skip void/None/null/undefined/empty
        if not return_type or return_type.lower() in ["void", "none", "null", "undefined", ""]:
            return context_docs

        # Skip primitives
        primitives = [
            "int",
            "str",
            "float",
            "bool",
            "double",
            "long",
            "short",
            "char",
            "byte",
            "boolean",
            "string",
            "number",
            "any",
            "object",
        ]
        if return_type.lower() in primitives:
            return context_docs

        # Clean generic/template types: List[User] → User, User[] → User, Optional[Token] → Token
        cleaned_type = return_type
        if "[" in cleaned_type:
            # Check if it's an array type (User[]) or generic (List[User])
            before_bracket = cleaned_type.split("[")[0].strip()
            after_bracket = cleaned_type.split("[")[1].rstrip("]").split(",")[0].strip()

            # If there's content after bracket, it's a generic (List[User])
            # If bracket content is empty, it's an array (User[])
            if after_bracket and after_bracket.lower() not in primitives:
                cleaned_type = after_bracket  # List[User] → User
            else:
                cleaned_type = before_bracket  # User[] → User
        elif "<" in cleaned_type:
            # C++ templates: Vector<User> → User
            inner = cleaned_type.split("<")[1].rstrip(">").split(",")[0].strip()
            cleaned_type = inner if inner and inner not in primitives else cleaned_type

        # Remove C++ pointer/reference markers: User* → User, const User& → User
        cleaned_type = cleaned_type.replace("*", "").replace("&", "").replace("const", "").strip()

        # Skip if we ended up with a primitive after cleaning
        if cleaned_type.lower() in primitives:
            return context_docs

        # Find return type class in graph
        file_path = node_data.get("file_path", "")
        language = node_data.get("language", "")
        return_type_node = self._find_graph_node(cleaned_type, file_path, language)

        if not return_type_node or return_type_node in processed_nodes:
            return context_docs

        # Only include architectural symbols (class, interface, struct, enum, module for C++, type_alias)
        # NOTE: C++ parser sometimes creates 'module' nodes for classes in header files
        return_type_data = self.graph.nodes.get(return_type_node, {})
        return_type_symbol_type = return_type_data.get("symbol_type", "").lower()

        if return_type_symbol_type not in ["class", "interface", "struct", "enum", "module", "type_alias"]:
            return context_docs

        # For type aliases, follow ALIAS_OF chain to bring underlying concrete types
        if return_type_symbol_type == "type_alias":
            resolved_targets = self._resolve_alias_chain(return_type_node)
            symbol_name = node_data.get("symbol_name", "")
            for target_node in resolved_targets:
                if target_node in processed_nodes:
                    continue
                target_data = self.graph.nodes.get(target_node, {})
                target_file = target_data.get("file_path", "")
                if not target_file or target_file.startswith("<") or "built-in" in target_file.lower():
                    continue
                target_doc = self._create_document_from_graph_node(
                    target_node,
                    expansion_reason="return_type_alias_target",
                    source_symbol=symbol_name,
                    via_context=f"via {symbol_name}() return type alias {cleaned_type}",
                )
                if target_doc:
                    if not existing_symbols or not self._is_symbol_already_retrieved(target_doc, existing_symbols):
                        processed_nodes.add(target_node)
                        target_name = target_data.get("symbol_name", "")
                        self.logger.debug(f"Expanded return type alias {cleaned_type} → {target_name}")
                        context_docs.append(target_doc)
            return context_docs

        # Create document for return type class
        symbol_name = node_data.get("symbol_name", "")
        return_type_doc = self._create_document_from_graph_node(
            return_type_node,
            expansion_reason="return_type_of",
            source_symbol=symbol_name,
            via_context=f"via {symbol_name}() return",
        )
        if return_type_doc:
            # Check if not already retrieved
            if not existing_symbols or not self._is_symbol_already_retrieved(return_type_doc, existing_symbols):
                processed_nodes.add(return_type_node)
                self.logger.debug(f"Expanded to return type: {cleaned_type} for {symbol_name}")
                context_docs.append(return_type_doc)

        return context_docs

    def _expand_parameter_types(
        self, node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand to parameter type classes using symbol metadata.

        Extracts parameter_types list from symbol metadata, cleans each type,
        and finds the type classes in graph.

        For functions: processData(data: DataFrame, config: Config) → bring DataFrame and Config classes
        For methods: Never executed (methods not in vector store)

        Args:
            node: Function node ID
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context (for deduplication)

        Returns:
            List of parameter type class documents
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(node):
            return context_docs

        node_data = self.graph.nodes.get(node, {})
        symbol = node_data.get("symbol")

        if not symbol or not hasattr(symbol, "parameter_types"):
            return context_docs

        parameter_types = symbol.parameter_types
        if not parameter_types:
            return context_docs

        file_path = node_data.get("file_path", "")
        language = node_data.get("language", "")

        # Primitives to skip
        primitives = [
            "int",
            "str",
            "float",
            "bool",
            "double",
            "long",
            "short",
            "char",
            "byte",
            "boolean",
            "string",
            "number",
            "any",
            "object",
            "void",
            "none",
        ]

        for param_type in parameter_types:
            if not param_type or param_type.lower() in primitives:
                continue

            # Clean generic/template types
            cleaned_type = param_type
            if "[" in cleaned_type:
                # Python/TypeScript arrays: User[] → User
                # Take the part before the bracket
                cleaned_type = cleaned_type.split("[")[0].strip()
            elif "<" in cleaned_type:
                # C++/Java generics: List<User> → User
                # Extract inner type
                inner = cleaned_type.split("<")[1].rstrip(">").split(",")[0].strip()
                cleaned_type = inner if inner and inner.lower() not in primitives else cleaned_type

            # Remove C++ pointer/reference markers
            cleaned_type = cleaned_type.replace("*", "").replace("&", "").replace("const", "").strip()

            if not cleaned_type or cleaned_type.lower() in primitives:
                continue

            # Find parameter type class
            param_type_node = self._find_graph_node(cleaned_type, file_path, language)

            if not param_type_node or param_type_node in processed_nodes:
                continue

            # Only architectural symbols (class, interface, struct, enum, module for C++, type_alias)
            # NOTE: C++ parser sometimes creates 'module' nodes for classes in header files
            param_type_data = self.graph.nodes.get(param_type_node, {})
            param_type_symbol_type = param_type_data.get("symbol_type", "").lower()

            if param_type_symbol_type not in ["class", "interface", "struct", "enum", "module", "type_alias"]:
                continue

            # Get source function name for transitive context
            func_name = node_data.get("symbol_name", "")

            # For type aliases, follow ALIAS_OF chain to bring underlying concrete types
            if param_type_symbol_type == "type_alias":
                resolved_targets = self._resolve_alias_chain(param_type_node)
                for target_node in resolved_targets:
                    if target_node in processed_nodes:
                        continue
                    target_data = self.graph.nodes.get(target_node, {})
                    target_file = target_data.get("file_path", "")
                    if not target_file or target_file.startswith("<") or "built-in" in target_file.lower():
                        continue
                    target_doc = self._create_document_from_graph_node(
                        target_node,
                        expansion_reason="param_type_alias_target",
                        source_symbol=func_name,
                        via_context=f"via {func_name}() param type alias {cleaned_type}",
                    )
                    if target_doc:
                        if not existing_symbols or not self._is_symbol_already_retrieved(target_doc, existing_symbols):
                            processed_nodes.add(target_node)
                            target_name = target_data.get("symbol_name", "")
                            self.logger.debug(f"Expanded param type alias {cleaned_type} → {target_name}")
                            context_docs.append(target_doc)
                continue

            param_type_doc = self._create_document_from_graph_node(
                param_type_node,
                expansion_reason="param_type",
                source_symbol=func_name,
                via_context=f"via {func_name}() parameter",
            )
            if param_type_doc:
                # Check if not already retrieved
                if not existing_symbols or not self._is_symbol_already_retrieved(param_type_doc, existing_symbols):
                    processed_nodes.add(param_type_node)
                    self.logger.debug(f"Expanded to parameter type: {cleaned_type}")
                    context_docs.append(param_type_doc)

        return context_docs

    def _expand_composed_classes_of_parameters(
        self,
        method_node: str,
        parameter_type_docs: list[Document],
        processed_nodes: set[str],
        existing_symbols: set[str] = None,
    ) -> list[Document]:
        """
        Expand to classes composed within parameter type classes (transitive field access).

        When a function has parameter `config: AppConfig`, and AppConfig has field `db: DatabaseConfig`,
        this brings DatabaseConfig into context to handle code like: `config.db.host`.

        This enables "transitive" field access expansion:
        function → parameter type → composed classes of parameter type

        Example:
            // C++ or Java
            class DatabaseConfig {
                string host;
            };

            class AppConfig {
                DatabaseConfig db;  // ← Composition edge created
            };

            void connect(AppConfig* config) {
                string host = config->db.host;  // ← Needs DatabaseConfig in context
            }

            Expansion result:
            - connect (function) ✅
            - AppConfig (parameter type) ✅
            - DatabaseConfig (composed class) ✅

        LANGUAGE SUPPORT STATUS (as of Nov 2025):
        ✅ C++: Full support
           - Creates composition edges: Class → DEFINES → Field → COMPOSITION → Type
           - Captures parameter types in symbol metadata
           - Transitive expansion works

        ✅ Java: Full support
           - Creates composition edges: Class → DEFINES → Field → COMPOSITION → Type
           - Captures parameter types in symbol metadata
           - Transitive expansion works

        ✅ Python: Full support (fixed Nov 2025)
           - Creates composition edges: Class → DEFINES → Field → COMPOSITION → Type
           - Captures parameter types from type hints
           - Transitive expansion works

        ⚠️ JavaScript: Partial support (acceptable)
           - Has: Composition, inheritance, calls, creates edges
           - Missing: Parameter/return types (would require JSDoc parsing)
           - Note: JSDoc usage is rare in practice - most typed JS projects use TypeScript
           - Impact: Low - expansion still works via calls/creates/composition edges

        Args:
            method_node: Function/method node ID
            parameter_type_docs: Documents for parameter type classes already expanded
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context

        Returns:
            List of composed class documents
        """
        context_docs = []

        if not self.graph or not parameter_type_docs:
            self.logger.debug(
                f"Skipping composed class expansion: graph={self.graph is not None}, param_docs={len(parameter_type_docs) if parameter_type_docs else 0}"
            )
            return context_docs

        self.logger.debug(f"Expanding composed classes for {len(parameter_type_docs)} parameter type(s)")

        # For each parameter type class, find its composed classes
        for param_doc in parameter_type_docs:
            param_node_id = param_doc.metadata.get("node_id")
            param_name = param_doc.metadata.get("symbol_name", "unknown")

            if not param_node_id or not self.graph.has_node(param_node_id):
                self.logger.debug(f"Skipping {param_name}: no valid node_id")
                continue

            param_data = self.graph.nodes.get(param_node_id, {})
            param_type = param_data.get("symbol_type", "").lower()

            self.logger.debug(f"Checking {param_name} (type: {param_type}) for composed classes")

            # Only expand composition for class-like types
            if param_type not in ["class", "interface", "struct"]:
                self.logger.debug(f"Skipping {param_name}: not a class-like type")
                continue

            # Find composition relationships (limit to 2 to avoid bloat)
            self.logger.debug(f"Finding composition for {param_name} (node: {param_node_id})")
            composed_docs = self._find_composition_context(param_node_id, processed_nodes, limit=2)

            # Filter out duplicates
            composed_docs = self._filter_unique_documents(composed_docs, existing_symbols)

            if composed_docs:
                param_name = param_data.get("symbol_name", "unknown")
                self.logger.debug(f"Expanded {len(composed_docs)} composed classes from parameter type {param_name}")
                context_docs.extend(composed_docs)

        return context_docs

    def _expand_created_classes(
        self, node: str, processed_nodes: set[str], existing_symbols: set[str] = None
    ) -> list[Document]:
        """
        Expand to classes that are constructed/created within a function/method.

        Looks for CREATES, REFERENCES, or USES_TYPE relationships from the function
        to classes, indicating object construction (e.g., new User(), User::create()).

        For Java/JavaScript methods, also checks parent class/module edges since
        relationships may be attached at class/module scope rather than method scope.

        For functions: createSession() { user = new User(); } → bring User class
        For methods: Never executed in production (methods not in vector store)

        Args:
            node: Function/method node ID
            processed_nodes: Set of already processed nodes
            existing_symbols: Set of symbols already in context (for deduplication)

        Returns:
            List of constructed class documents
        """
        context_docs = []

        if not self.graph or not self.graph.has_node(node):
            return context_docs

        node_data = self.graph.nodes.get(node, {})

        # Collect nodes to check for edges (method + parent class/module for Java/JavaScript)
        nodes_to_check = [node]

        # For Java/JavaScript methods, also check parent class/module edges
        # Use DEFINES edges to find parent (O(degree) instead of O(N))
        # Week 7: All containers now create DEFINES edges to their children
        if node_data.get("symbol_type") == "method":
            for pred in self.graph.predecessors(node):
                edge_data_pred = self.graph.get_edge_data(pred, node)
                if edge_data_pred:
                    for edge_pred in edge_data_pred.values():
                        rel_type_pred = edge_pred.get("relationship_type", "").lower()
                        if rel_type_pred == "defines":
                            # Found parent via DEFINES edge
                            pred_data = self.graph.nodes.get(pred, {})
                            pred_type = pred_data.get("symbol_type", "")
                            if pred_type in ["class", "module", "namespace"]:
                                nodes_to_check.append(pred)
                                break

        # Find outgoing CREATES, REFERENCES, or USES_TYPE relationships
        # These indicate object construction or type usage
        for check_node in nodes_to_check:
            for _, target_node, edge_data in self.graph.out_edges(check_node, data=True):
                edge_type = edge_data.get("relationship_type", "").lower()

                # Look for construction/creation relationships
                if edge_type not in ["creates", "references", "uses_type"]:
                    continue

                # Check if target is a class/struct/interface (architectural symbol)
                target_data = self.graph.nodes.get(target_node, {})
                target_symbol_type = target_data.get("symbol_type", "").lower()

                # Include architectural types + functions (for React JSX component references)
                # Functions included because JSX elements like <UserCard /> create 'references' edges to component functions
                # Note: 'constant' included for JavaScript/TypeScript imported classes (e.g., const User = require('./User'))
                if target_symbol_type not in ["class", "interface", "struct", "enum", "module", "constant", "function"]:
                    continue

                # Skip if already processed
                if target_node in processed_nodes:
                    continue

                # Create document for constructed class
                class_doc = self._create_document_from_graph_node(target_node)
                if class_doc:
                    # Check if not already retrieved
                    if not existing_symbols or not self._is_symbol_already_retrieved(class_doc, existing_symbols):
                        processed_nodes.add(target_node)
                        class_name = target_data.get("symbol_name", "unknown")
                        self.logger.debug(
                            f"Expanded to created class/function: {class_name} from {node_data.get('symbol_name')}"
                        )
                        context_docs.append(class_doc)
                        # For functions (like React components), recursively expand to get their references
                        if target_symbol_type == "function":
                            function_expanded = self._expand_created_classes(
                                target_node, processed_nodes, existing_symbols
                            )
                            context_docs.extend(function_expanded)

        return context_docs

    def _find_parent_context_for_method(
        self, method_node: str, processed_nodes: set[str], original_doc: Document = None
    ) -> list[Document]:
        """
        Find parent context for a method using both code_graph and document metadata.

        When multiple methods with the same name exist (e.g., start method in Engine, Vehicle, Car),
        the code_graph may have a single unified node. Use the original document's metadata to
        determine which specific class contains this method instance.
        """
        parent_docs = []

        if not self.graph or not self.graph.has_node(method_node):
            return parent_docs

        # Strategy 1: Use original document metadata to find containing class
        if original_doc and original_doc.metadata:
            metadata = original_doc.metadata
            file_path = metadata.get("file_path", "")
            language = metadata.get("language", "")

            # Look for containing class information in the document content or nearby lines
            content = original_doc.page_content
            start_line = metadata.get("start_line", 0)

            # Try to infer containing class from content structure
            containing_class = self._infer_containing_class_from_content(content, file_path, start_line)
            if containing_class:
                # Find the class node in the code_graph
                class_node = self._find_graph_node(containing_class, file_path, language)
                if class_node and class_node not in processed_nodes:
                    class_doc = self._create_document_from_graph_node(class_node)
                    if class_doc:
                        parent_docs.append(class_doc)
                        processed_nodes.add(class_node)
                        self.logger.debug(f"Found containing class from metadata: {containing_class}")
                        return parent_docs

        # Strategy 2: Fall back to original parent finding logic
        if not parent_docs:
            parent_docs = self._find_parent_context(method_node, processed_nodes)

        return parent_docs

    def _infer_containing_class_from_content(self, content: str, file_path: str, start_line: int) -> str:
        """
        Infer the containing class from method content and file structure.

        For Python, look for class definition patterns above the method.
        """
        if not content:
            return None

        # For Python methods, look for class definition pattern
        content.split("\n")

        # If this is just the method content, we need to look at file structure
        # Look for class keyword in the content or use heuristics

        # Simple heuristic: if method calls self.something, it's an instance method
        if "self." in content:
            # Try to read the file and find the class containing this method
            try:
                if file_path and os.path.exists(file_path):
                    with open(file_path) as f:
                        file_lines = f.readlines()

                    # Find the class that contains the method at start_line
                    current_class = None
                    for i, line in enumerate(file_lines):
                        line_num = i + 1
                        if line_num >= start_line:
                            break

                        stripped = line.strip()
                        if stripped.startswith("class ") and ":" in stripped:
                            # Extract class name
                            class_def = stripped[6:].split(":")[0].strip()
                            if "(" in class_def:
                                current_class = class_def.split("(")[0].strip()
                            else:
                                current_class = class_def

                    return current_class
            except Exception:  # noqa: S110
                pass

        return None


def expand_content_simple(documents: list[Document]) -> list[Document]:
    """
    Simple content expansion function that can be used without a code_graph store.

    Args:
        documents: List of retrieved documents

    Returns:
        List of documents with basic deduplication applied
    """
    expander = ContentExpander()
    return expander.expand_retrieved_documents(documents)
