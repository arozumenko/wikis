"""
Document Ranking Module for Hierarchical Context Generation.

This module ranks expanded documents using comprehensive graph relationship analysis
to determine which documents should be in Tier 1 (full code), Tier 2 (signatures),
or Tier 3 (summaries).

Ranking Factors:
- Page-specified locations (target_folders, key_files)
- Containment relationships (defines, contains, has_member)
- Call relationships (calls, invokes, instantiates, creates)
- Inheritance relationships (inheritance, extends, implements, implementation)
- Composition relationships (uses, composition, aggregation, references)
- Symbol type importance (class > enum > function)
- Module proximity (same module/package)
- Graph distance penalties
"""

import logging
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document

from .token_counter import get_token_counter

logger = logging.getLogger(__name__)


class DocumentRanker:
    """Ranks documents based on relevance using graph relationships."""

    # Relationship type weights (higher = more important)
    RELATIONSHIP_WEIGHTS = {
        # Containment relationships - strongest signal
        "defines": 15,
        "contains": 15,
        "has_member": 15,
        # Call relationships - direct runtime dependencies
        "calls": 12,
        "invokes": 12,
        "instantiates": 12,
        "creates": 12,
        # Inheritance relationships - structural dependencies
        "inheritance": 10,
        "extends": 10,
        "implements": 10,
        "implementation": 10,
        # Composition relationships - data flow dependencies
        "uses": 8,
        "composition": 8,
        "aggregation": 8,
        "references": 8,
    }

    # Symbol type importance scores
    # Note: Structs (C++, Rust, Go) and Traits (Rust) are equivalent to classes
    # Documentation files are HIGH priority - they already contain human-readable explanations
    SYMBOL_TYPE_SCORES = {
        # Architectural types - highest priority
        "class": 10,
        "interface": 10,
        "struct": 10,  # C++, Rust, Go structs
        "trait": 10,  # Rust traits
        "protocol": 10,  # Swift protocols
        # Documentation files - critical for context
        "readme": 10,  # README files explain entire system
        "documentation": 9,  # API docs, design docs, architecture docs
        "markdown": 8,  # General markdown documentation
        "text_chunk": 8,  # Documentation text chunks
        "config": 8,  # Configuration files (package.json, .toml, .yaml)
        # Implementation types
        "impl": 9,  # Rust impl blocks, C++ struct::method implementations
        "enum": 8,
        # Functional types
        "type_alias": 7,  # TypeScript type aliases, Rust type definitions
        "function": 7,
        "constant": 7,  # Constants define system behavior boundaries
        # Supporting types
        "method": 6,
        "namespace": 5,  # C++, C# namespaces
        "module": 5,  # Rust, Python modules
        "variable": 3,
        "macro": 3,  # Rust macros, C++ macros
    }

    # Token budgets per tier
    TIER_BUDGETS = {
        1: 50000,  # Full code
        2: 30000,  # Signatures (with compression)
        3: 10000,  # Summaries (with compression)
    }

    # Compression ratios (used for tier assignment)
    COMPRESSION_RATIOS = {
        1: 1.0,  # No compression
        2: 0.35,  # ~35% of original
        3: 0.10,  # ~10% of original
    }

    def __init__(self, code_graph: nx.Graph):
        """
        Initialize the document ranker.

        Args:
            code_graph: NetworkX graph with code relationships
        """
        self.code_graph = code_graph

    def rank_expanded_documents(
        self, page_spec: dict, expanded_docs: list[Document], total_token_budget: int = 90000
    ) -> tuple[list[tuple[Document, int, float]], dict]:
        """
        Rank expanded documents and assign tiers based on relevance.

        Args:
            page_spec: Page specification with:
                - topic: Main topic/symbol name
                - target_folders: List of folder paths to prioritize
                - key_files: List of file patterns to prioritize
            expanded_docs: List of 100-140 expanded documents from ContentExpander
            total_token_budget: Total tokens available (default 90K)

        Returns:
            Tuple of:
                - List of (Document, tier, score) sorted by tier then score
                - Metrics dict with tier distributions and token usage
        """
        if not expanded_docs:
            return [], self._empty_metrics()

        logger.info(f"Ranking {len(expanded_docs)} expanded documents for page: {page_spec.get('topic', 'unknown')}")

        # Step 1: Score all documents
        scored_docs = []
        for doc in expanded_docs:
            score = self._calculate_document_score(doc, page_spec)
            scored_docs.append((doc, score))

        # Step 2: Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Assign tiers based on scores and token budgets
        tiered_docs, metrics = self._assign_tiers(scored_docs, total_token_budget)

        logger.info(
            f"Ranking complete: Tier1={metrics['tier1_count']}, "
            f"Tier2={metrics['tier2_count']}, Tier3={metrics['tier3_count']}, "
            f"Total tokens={metrics['total_tokens']:.0f}"
        )

        return tiered_docs, metrics

    def _calculate_document_score(self, doc: Document, page_spec: dict) -> float:
        """
        Calculate relevance score for a document.

        Scoring factors:
        1. Page-specified locations (target_folders, key_files): +20
        2. Relationship types (containment/calls/inheritance/composition): +15/12/10/8
        3. Symbol type importance (class > enum > function): +10 to +3
        4. Module proximity (same module/package): +8/5
        5. Graph distance penalty: -0.5 per hop
        6. File size penalty: -1 to -2 for very large files

        Args:
            doc: Document to score
            page_spec: Page specification

        Returns:
            Float score (higher = more relevant)
        """
        score = 0.0
        metadata = doc.metadata

        # Extract document identity
        file_path = metadata.get("file_path", "")
        symbol_name = metadata.get("symbol_name", "")
        symbol_type = metadata.get("symbol_type", "").lower()

        # Factor 1: Page-specified locations (highest priority)
        if self._is_in_target_locations(file_path, page_spec):
            score += 20
            logger.debug(f"  {symbol_name}: +20 (target location)")

        # Factor 2: Relationship scores
        relationship_score = self._calculate_relationship_score(symbol_name, page_spec.get("topic", ""))
        score += relationship_score
        if relationship_score > 0:
            logger.debug(f"  {symbol_name}: +{relationship_score:.1f} (relationships)")

        # Factor 3: Symbol type importance
        type_score = self.SYMBOL_TYPE_SCORES.get(symbol_type, 1)
        score += type_score
        logger.debug(f"  {symbol_name}: +{type_score} (type={symbol_type})")

        # Factor 4: Module proximity
        proximity_score = self._calculate_proximity_score(file_path, page_spec.get("topic", ""))
        score += proximity_score
        if proximity_score > 0:
            logger.debug(f"  {symbol_name}: +{proximity_score} (proximity)")

        # Factor 5: Graph distance penalty
        distance_penalty = self._calculate_distance_penalty(symbol_name, page_spec.get("topic", ""))
        score += distance_penalty
        if distance_penalty < 0:
            logger.debug(f"  {symbol_name}: {distance_penalty:.1f} (distance penalty)")

        # Factor 6: File size penalty (discourage very large files)
        size_penalty = self._calculate_size_penalty(doc.page_content)
        score += size_penalty
        if size_penalty < 0:
            logger.debug(f"  {symbol_name}: {size_penalty:.1f} (size penalty)")

        logger.debug(f"Final score for {symbol_name}: {score:.1f}")
        return score

    def _is_in_target_locations(self, file_path: str, page_spec: dict) -> bool:
        """Check if document is in target folders or matches key files."""
        target_folders = page_spec.get("target_folders", [])
        key_files = page_spec.get("key_files", [])

        file_path_obj = Path(file_path)

        # Check target folders
        for folder in target_folders:
            if folder in file_path:
                return True

        # Check key file patterns
        for pattern in key_files:
            if pattern in file_path or file_path_obj.match(pattern):
                return True

        return False

    def _calculate_relationship_score(self, symbol_name: str, topic: str) -> float:
        """
        Calculate score based on graph relationships to the topic.

        Uses ALL 14 relationship types with appropriate weights.
        """
        if not self.code_graph or not topic or not symbol_name:
            return 0.0

        # Check if both nodes exist in graph
        if symbol_name not in self.code_graph or topic not in self.code_graph:
            return 0.0

        score = 0.0

        # Get all edges between symbol and topic (both directions)
        # NOTE: For MultiDiGraph, get_edge_data returns {edge_key: {data}, ...}
        # Each edge has 'relationship_type' (singular) as the key
        edges_to_topic = self.code_graph.get_edge_data(symbol_name, topic)
        edges_from_topic = self.code_graph.get_edge_data(topic, symbol_name)

        # Score edges to topic (symbol -> topic)
        if edges_to_topic:
            # MultiDiGraph returns dict of edge_key -> edge_data
            for _edge_key, edge_data in edges_to_topic.items():
                rel_type = edge_data.get("relationship_type", "")
                weight = self.RELATIONSHIP_WEIGHTS.get(rel_type, 0)
                score += weight

        # Score edges from topic (topic -> symbol)
        if edges_from_topic:
            # MultiDiGraph returns dict of edge_key -> edge_data
            for _edge_key, edge_data in edges_from_topic.items():
                rel_type = edge_data.get("relationship_type", "")
                weight = self.RELATIONSHIP_WEIGHTS.get(rel_type, 0)
                score += weight

        return score

    def _calculate_proximity_score(self, file_path: str, topic: str) -> float:
        """
        Calculate score based on module/package proximity.

        Same module: +8
        Same package: +5
        Different package: +0
        """
        if not topic or not file_path:
            return 0.0

        # Try to find topic node in graph to get its file path
        topic_file_path = None
        if self.code_graph and topic in self.code_graph:
            topic_node_data = self.code_graph.nodes[topic]
            topic_file_path = topic_node_data.get("file_path", "")

        if not topic_file_path:
            return 0.0

        file_path_obj = Path(file_path)
        topic_path_obj = Path(topic_file_path)

        # Same file = same module
        if file_path == topic_file_path:
            return 8

        # Same parent directory = same package
        if file_path_obj.parent == topic_path_obj.parent:
            return 5

        return 0.0

    def _calculate_distance_penalty(self, symbol_name: str, topic: str) -> float:
        """
        Calculate penalty based on graph distance from topic.

        Penalty: -0.5 per hop
        """
        if not self.code_graph or not topic or not symbol_name:
            return 0.0

        if symbol_name not in self.code_graph or topic not in self.code_graph:
            return 0.0

        try:
            # Calculate shortest path length
            distance = nx.shortest_path_length(self.code_graph, source=topic, target=symbol_name)
            return -0.5 * distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists - apply maximum penalty
            return -5.0

    def _calculate_size_penalty(self, content: str) -> float:
        """
        Calculate penalty for very large files.

        Rationale: Very large files might be generated code or
        have too much detail for Tier 1.

        < 5000 chars: no penalty
        5000-10000 chars: -1
        > 10000 chars: -2
        """
        length = len(content)

        if length < 5000:
            return 0.0
        elif length < 10000:
            return -1.0
        else:
            return -2.0

    def _assign_tiers(
        self, scored_docs: list[tuple[Document, float]], total_token_budget: int
    ) -> tuple[list[tuple[Document, int, float]], dict]:
        """
        Assign tiers to documents based on scores and token budgets.

        Strategy:
        1. Iterate through sorted documents (highest score first)
        2. Try to assign to Tier 1 if budget allows
        3. If Tier 1 full, try Tier 2 (account for compression)
        4. If Tier 2 full, try Tier 3 (account for compression)
        5. If all tiers full, skip document

        Args:
            scored_docs: List of (Document, score) tuples sorted by score
            total_token_budget: Total tokens available

        Returns:
            Tuple of (tiered_docs, metrics)
        """
        tier_assignments = []
        tier_tokens = {1: 0.0, 2: 0.0, 3: 0.0}
        tier_budgets = self.TIER_BUDGETS.copy()

        for doc, score in scored_docs:
            doc_tokens = self._estimate_tokens(doc.page_content)
            assigned = False

            # Try Tier 1 first (highest relevance, full code)
            if tier_tokens[1] + doc_tokens <= tier_budgets[1]:
                tier_assignments.append((doc, 1, score))
                tier_tokens[1] += doc_tokens
                assigned = True

            # Try Tier 2 (medium relevance, signatures only)
            elif tier_tokens[2] + (doc_tokens * self.COMPRESSION_RATIOS[2]) <= tier_budgets[2]:
                tier_assignments.append((doc, 2, score))
                tier_tokens[2] += doc_tokens * self.COMPRESSION_RATIOS[2]
                assigned = True

            # Try Tier 3 (lower relevance, summaries only)
            elif tier_tokens[3] + (doc_tokens * self.COMPRESSION_RATIOS[3]) <= tier_budgets[3]:
                tier_assignments.append((doc, 3, score))
                tier_tokens[3] += doc_tokens * self.COMPRESSION_RATIOS[3]
                assigned = True

            if not assigned:
                logger.warning(
                    f"Document {doc.metadata.get('symbol_name', 'unknown')} exceeds all tier budgets - skipping"
                )

        # Sort by tier (Tier 1 first) then by score within tier
        tier_assignments.sort(key=lambda x: (x[1], -x[2]))

        # Calculate metrics
        metrics = {
            "tier1_count": sum(1 for t in tier_assignments if t[1] == 1),
            "tier2_count": sum(1 for t in tier_assignments if t[1] == 2),
            "tier3_count": sum(1 for t in tier_assignments if t[1] == 3),
            "tier1_tokens": tier_tokens[1],
            "tier2_tokens": tier_tokens[2],
            "tier3_tokens": tier_tokens[3],
            "total_tokens": sum(tier_tokens.values()),
            "total_docs": len(tier_assignments),
            "skipped_docs": len(scored_docs) - len(tier_assignments),
        }

        return tier_assignments, metrics

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses centralized TokenCounter with tiktoken (accurate) or fallback estimation.
        """
        return get_token_counter().count(text)

    def _empty_metrics(self) -> dict:
        """Return empty metrics dict."""
        return {
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "tier1_tokens": 0.0,
            "tier2_tokens": 0.0,
            "tier3_tokens": 0.0,
            "total_tokens": 0.0,
            "total_docs": 0,
            "skipped_docs": 0,
        }


def rank_expanded_documents(
    page_spec: dict, expanded_docs: list[Document], code_graph: nx.Graph, total_token_budget: int = 90000
) -> tuple[list[tuple[Document, int, float]], dict]:
    """
    Convenience function to rank expanded documents.

    Args:
        page_spec: Page specification with topic, target_folders, key_files
        expanded_docs: List of expanded documents from ContentExpander
        code_graph: NetworkX graph with code relationships
        total_token_budget: Total tokens available (default 90K)

    Returns:
        Tuple of (tiered_docs, metrics)
        - tiered_docs: List of (Document, tier, score) sorted by tier then score
        - metrics: Dict with tier counts and token usage
    """
    ranker = DocumentRanker(code_graph)
    return ranker.rank_expanded_documents(page_spec, expanded_docs, total_token_budget)
