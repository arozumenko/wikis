"""
Retriever stack with ensemble retrievers and cross-encoder reranking
"""

import logging
import os
from typing import Any

from langchain_core.documents import Document

# Use langchain_classic for retrievers/rerankers while LangChain v1+ migrates APIs.
try:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    logging.warning("langchain_classic retrievers not available")
    ContextualCompressionRetriever = None
    EnsembleRetriever = None

try:
    from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
    from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
except ImportError:
    BaseCrossEncoder = None
    CrossEncoderReranker = None
    logging.warning("langchain_classic CrossEncoderReranker not available")

# EmbeddingsFilter for semantic doc reranking
try:
    from langchain_classic.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter

    EMBEDDINGS_FILTER_AVAILABLE = True
except ImportError:
    EmbeddingsFilter = None
    EMBEDDINGS_FILTER_AVAILABLE = False
    logging.warning("langchain_classic EmbeddingsFilter not available")

try:
    import langchain_community.retrievers  # noqa: F401 — availability check only

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("BM25Retriever not available")

try:
    from langchain_community.tools.tavily_search import TavilySearchResults

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("TavilySearchResults not available")

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("CrossEncoder not available")

import networkx as nx

from .content_expander import ContentExpander
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flags for Doc/Code Separation
# =============================================================================
# Use semantic retrieval (Ensemble + EmbeddingsFilter) for docs
USE_SEMANTIC_DOC_RETRIEVAL = os.getenv("WIKIS_DOC_SEMANTIC_RETRIEVAL", "0") == "1"

# Documentation symbol types — single source of truth in constants.py
from .constants import DOC_CHUNK_TYPES, DOC_SYMBOL_TYPES, DOCUMENTATION_EXTENSIONS_SET, KNOWN_FILENAMES

if USE_SEMANTIC_DOC_RETRIEVAL:
    logger.info("[FEATURE FLAG] WIKIS_DOC_SEMANTIC_RETRIEVAL=1: Semantic doc retrieval enabled in retriever stack")


class SentenceTransformersCrossEncoderAdapter(BaseCrossEncoder if BaseCrossEncoder else object):
    """Adapter to satisfy langchain_classic BaseCrossEncoder using sentence-transformers."""

    def __init__(self, model_name: str, cache_folder: str | None = None, device: str = "cpu"):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers CrossEncoder is not available")
        self._model = CrossEncoder(model_name, device=device, cache_folder=cache_folder)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(text_pairs)
        # predict may return numpy array; normalize to plain floats.
        return [float(s) for s in list(scores)]


class WebRetriever:
    """Web search retriever using Tavily"""

    def __init__(self, k: int = 10, api_key: str | None = None):
        self.k = k
        self.api_key = api_key
        self.search_tool = None

        if TAVILY_AVAILABLE and api_key:
            try:
                self.search_tool = TavilySearchResults(max_results=k, search_depth="advanced", api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily search: {e}")

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Get web search results as documents"""
        if not self.search_tool:
            logger.debug("Web search tool not available")
            return []

        try:
            results = self.search_tool.run(query)
            documents = []

            if isinstance(results, list):
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        url = result.get("url", f"web_result_{i}")
                        title = result.get("title", "Web Result")

                        doc = Document(
                            page_content=content,
                            metadata={"source": url, "title": title, "chunk_type": "web", "search_query": query},
                        )
                        documents.append(doc)

            logger.debug(f"Retrieved {len(documents)} web search results")
            return documents

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []


class WikiRetrieverStack:
    """Complete retriever stack with ensemble and reranking"""

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        relationship_graph: nx.DiGraph | None = None,
        tavily_api_key: str | None = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_client=None,
        use_enhanced_graph: bool = True,
        graph_text_index=None,
    ):

        self.vectorstore_manager = vectorstore_manager
        self.relationship_graph = relationship_graph
        self.tavily_api_key = tavily_api_key
        self.cross_encoder_model = cross_encoder_model
        self.llm_client = llm_client
        self.use_enhanced_graph = use_enhanced_graph

        # Initialize content expander for intelligent post-processing
        self.content_expander = ContentExpander(
            relationship_graph,
            graph_text_index=graph_text_index,
        )

        # Core retrievers
        self.dense_retriever = None
        self.bm25_retriever = None
        self.graph_retriever = None
        self.web_retriever = None

        # Ensemble retrievers
        self.repo_retriever = None
        self.research_retriever = None

        # Reranker
        self.reranker = None

        # Initialize components
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize all retriever components"""
        try:
            # Get vector store and documents
            vectorstore = self.vectorstore_manager.get_vectorstore()
            all_documents = self.vectorstore_manager.get_all_documents()

            if not vectorstore or not all_documents:
                logger.warning("No vector store or documents available")
                return

            # 1. Dense retriever (FAISS) - reduced k since we now store only architectural symbols
            self.dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})
            logger.info("Initialized dense retriever")

            # 2. BM25 retriever - use disk-backed mmap only (no in-memory fallback to prevent RAM explosion)
            if all_documents:
                try:
                    mmap_bm25 = self.vectorstore_manager.get_mmap_bm25_retriever(k=25)
                    if mmap_bm25:
                        self.bm25_retriever = mmap_bm25
                        logger.info("Initialized mmap BM25 retriever")
                    else:
                        # Skip BM25 entirely if disk-backed version unavailable
                        # In-memory BM25 would load all documents into RAM
                        logger.warning("Disk-backed BM25 unavailable, skipping BM25 retriever (dense retrieval only)")
                except Exception as e:
                    logger.warning(f"Failed to initialize BM25 retriever: {e}")

            # 5. Cross-encoder reranker (disabled by default for performance)
            # Set WIKIS_ENABLE_RERANKER=1 to enable
            enable_reranker = os.getenv("WIKIS_ENABLE_RERANKER") == "1"
            if (
                enable_reranker
                and CROSS_ENCODER_AVAILABLE
                and CrossEncoderReranker is not None
                and BaseCrossEncoder is not None
            ):
                try:
                    rerank_top_n = 25
                    env_top_n = os.getenv("WIKIS_RERANK_TOP_N")
                    if env_top_n:
                        try:
                            rerank_top_n = int(env_top_n)
                        except ValueError:
                            logger.warning(f"Invalid WIKIS_RERANK_TOP_N='{env_top_n}', using default 25")

                    ce_model = SentenceTransformersCrossEncoderAdapter(
                        model_name=self.cross_encoder_model,
                        device="cpu",
                    )
                    self.reranker = CrossEncoderReranker(
                        model=ce_model,
                        top_n=rerank_top_n,  # Reranker returns top N docs for expansion
                    )
                    logger.info(f"Initialized cross-encoder reranker: {self.cross_encoder_model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize cross-encoder reranker: {e}")

            # Build ensemble retrievers
            self._build_ensemble_retrievers()

        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")

    def _build_ensemble_retrievers(self):
        """Build ensemble retrievers with simplified architecture"""
        try:
            # Repository retriever - flatten all retrievers into one ensemble
            all_retrievers = []
            all_weights = []

            if self.dense_retriever:
                all_retrievers.append(self.dense_retriever)
                all_weights.append(0.6)

            if self.bm25_retriever:
                all_retrievers.append(self.bm25_retriever)
                all_weights.append(0.4)

            # Create single flat ensemble instead of nested ones
            if len(all_retrievers) >= 2:
                self.repo_retriever = EnsembleRetriever(retrievers=all_retrievers, weights=all_weights)
                logger.info(f"Built flat ensemble with {len(all_retrievers)} retrievers")

                if self.reranker:
                    self.repo_retriever = ContextualCompressionRetriever(
                        base_retriever=self.repo_retriever, base_compressor=self.reranker
                    )

            elif len(all_retrievers) == 1:
                # If only one retriever, use it directly
                self.repo_retriever = all_retrievers[0]
                logger.info("Using single retriever (no ensemble needed)")
            else:
                logger.warning("No retrievers available for ensemble")
                return

        except Exception as e:
            logger.error(f"Failed to build ensemble retrievers: {e}")
            # Fallback to dense retriever only
            if self.dense_retriever:
                self.repo_retriever = self.dense_retriever
                self.research_retriever = self.dense_retriever
                logger.info("Fallback: Using dense retriever only")

    def build_repo_retriever(self, k: int = 20) -> Any | None:
        """
        Build repository-focused retriever

        Returns retriever configured for repository-only search
        """
        if not self.repo_retriever:
            logger.warning("Repository retriever not available")
            return None

        # Configure k parameter
        if hasattr(self.repo_retriever, "search_kwargs"):
            self.repo_retriever.search_kwargs = {"k": k}

        return self.repo_retriever

    def search_repository(self, query: str, k: int = 15, apply_expansion: bool = True) -> list[Document]:
        """
        Retriever with code_graph expansion - reduced default k for architectural symbols.

        NOTE: k parameter controls initial retrieval, but ContentExpander may return
        MORE documents (100-140) by following relationships. This is intentional -
        the expanded set provides comprehensive context for wiki generation.
        DO NOT truncate expanded results here. Let wiki_graph_optimized decide
        how to handle large document sets (simple mode vs hierarchical mode).
        """
        # Best-effort: push k into the underlying retriever if it supports search_kwargs.
        # This keeps the public API stable while allowing callers to control initial breadth.
        try:
            if hasattr(self.repo_retriever, "search_kwargs"):
                self.repo_retriever.search_kwargs = {"k": k}
            elif hasattr(self.repo_retriever, "base_retriever") and hasattr(
                self.repo_retriever.base_retriever, "search_kwargs"
            ):
                self.repo_retriever.base_retriever.search_kwargs = {"k": k}
        except Exception:  # noqa: S110
            # If the retriever doesn't expose search_kwargs, we still proceed.
            pass

        raw_result = self.repo_retriever.invoke(query)

        # Mark initially retrieved docs (before expansion) for hybrid hints
        # Initial docs get forward hints (→), expanded docs get backward hints (←)
        if raw_result:
            for doc in raw_result:
                doc.metadata["is_initially_retrieved"] = True
            logger.info(f"[RETRIEVAL] Marked {len(raw_result)} documents with is_initially_retrieved=True")

        if not apply_expansion:
            # Return only initially retrieved docs.
            return raw_result if raw_result else []

        # Apply content expansion even in streamlit safe mode
        expanded_result = self.content_expander.expand_retrieved_documents(raw_result or [])

        # Log how many documents retained the flag after expansion
        if expanded_result:
            initially_retrieved_count = sum(
                1 for doc in expanded_result if doc.metadata.get("is_initially_retrieved", False)
            )
            logger.info(
                f"[RETRIEVAL] After expansion: {len(expanded_result)} docs, "
                f"{initially_retrieved_count} with is_initially_retrieved=True"
            )

        # Return ALL expanded documents - do not truncate to k
        # Wiki generation will handle large document sets intelligently
        return expanded_result if expanded_result else []

    def search_docs_semantic(self, query: str, k: int = 10, similarity_threshold: float = 0.7) -> list[Document]:
        """
        Semantic documentation retrieval using Ensemble + EmbeddingsFilter reranking.

        This method retrieves documentation files (README, docs/*.md, etc.) and reranks
        them by semantic similarity to the query using EmbeddingsFilter.

        Used when WIKIS_DOC_SEMANTIC_RETRIEVAL=1 for:
        - Wiki page generation (Related Documentation section)
        - Ask tool (documentation context)
        - Deep Research (background documentation)

        Args:
            query: Search query describing the topic
            k: Maximum number of documents to return after reranking
            similarity_threshold: Minimum similarity score for inclusion (0.0-1.0)

        Returns:
            List of documentation Documents, semantically reranked
        """
        if not query or not query.strip():
            logger.warning("[SEMANTIC_DOC_RETRIEVAL] Empty query provided")
            return []

        # Check if retriever is available
        if not self.repo_retriever:
            logger.warning("[SEMANTIC_DOC_RETRIEVAL] No retriever available")
            return []

        try:
            # Step 1: Get candidates from ensemble retriever (Dense + BM25)
            # Use higher k for initial retrieval to have good candidates for reranking
            initial_k = min(k * 5, 50)

            try:
                if hasattr(self.repo_retriever, "search_kwargs"):
                    self.repo_retriever.search_kwargs = {"k": initial_k}
                elif hasattr(self.repo_retriever, "base_retriever") and hasattr(
                    self.repo_retriever.base_retriever, "search_kwargs"
                ):
                    self.repo_retriever.base_retriever.search_kwargs = {"k": initial_k}
            except Exception:  # noqa: S110
                pass

            # Safely invoke the retriever with fallback to dense-only
            raw_candidates = []
            try:
                raw_candidates = self.repo_retriever.invoke(query)
            except Exception as retriever_error:
                error_msg = str(retriever_error)
                logger.warning(
                    f"[SEMANTIC_DOC_RETRIEVAL] Ensemble retriever failed: {error_msg}, trying dense retriever"
                )
                # Fallback to dense retriever only (skip BM25)
                if self.dense_retriever:
                    try:
                        if hasattr(self.dense_retriever, "search_kwargs"):
                            self.dense_retriever.search_kwargs = {"k": initial_k}
                        raw_candidates = self.dense_retriever.invoke(query)
                        logger.info(
                            f"[SEMANTIC_DOC_RETRIEVAL] Dense retriever fallback returned {len(raw_candidates)} candidates"
                        )
                    except Exception as dense_error:
                        logger.warning(f"[SEMANTIC_DOC_RETRIEVAL] Dense retriever also failed: {dense_error}")
                        return []
                else:
                    return []

            if not raw_candidates:
                logger.info(f"[SEMANTIC_DOC_RETRIEVAL] No candidates from retriever for query: '{query[:50]}...'")
                return []

            # Step 2: Filter to documentation types only
            doc_candidates = []
            for doc in raw_candidates:
                chunk_type = doc.metadata.get("chunk_type", "").lower()
                symbol_type = doc.metadata.get("symbol_type", "").lower()
                source = doc.metadata.get("source", "").lower()

                is_doc = (
                    chunk_type in DOC_CHUNK_TYPES
                    or symbol_type in DOC_SYMBOL_TYPES
                    or any(source.endswith(ext) for ext in DOCUMENTATION_EXTENSIONS_SET)
                    or source.rsplit("/", 1)[-1] in KNOWN_FILENAMES
                    or "/docs/" in source
                    or "readme" in source.rsplit("/", 1)[-1].lower()
                )

                if is_doc:
                    doc.metadata["is_documentation"] = True
                    doc.metadata["semantic_retrieved"] = True
                    doc_candidates.append(doc)

            if not doc_candidates:
                logger.info(f"[SEMANTIC_DOC_RETRIEVAL] No doc-type documents in {len(raw_candidates)} candidates")
                return []

            logger.info(
                f"[SEMANTIC_DOC_RETRIEVAL] Found {len(doc_candidates)} doc candidates from {len(raw_candidates)} total"
            )

            # Step 3: Apply EmbeddingsFilter for semantic reranking
            if USE_SEMANTIC_DOC_RETRIEVAL and EMBEDDINGS_FILTER_AVAILABLE and EmbeddingsFilter:
                try:
                    embeddings = self.vectorstore_manager.embeddings
                    logger.debug(f"[SEMANTIC_DOC_RETRIEVAL] Embeddings type: {type(embeddings)}")
                    if embeddings:
                        embeddings_filter = EmbeddingsFilter(
                            embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
                        )
                        logger.debug(
                            f"[SEMANTIC_DOC_RETRIEVAL] Calling compress_documents with {len(doc_candidates)} docs"
                        )
                        reranked = embeddings_filter.compress_documents(documents=doc_candidates, query=query)
                        doc_candidates = list(reranked)
                        logger.info(
                            f"[SEMANTIC_DOC_RETRIEVAL] EmbeddingsFilter reranked to {len(doc_candidates)} docs "
                            f"(threshold={similarity_threshold})"
                        )
                    else:
                        logger.warning("[SEMANTIC_DOC_RETRIEVAL] No embeddings available for EmbeddingsFilter")
                        doc_candidates = doc_candidates[:k]
                except Exception as e:
                    import traceback

                    logger.warning(f"[SEMANTIC_DOC_RETRIEVAL] EmbeddingsFilter failed: {e}")
                    logger.warning(f"[SEMANTIC_DOC_RETRIEVAL] EmbeddingsFilter traceback:\n{traceback.format_exc()}")
                    doc_candidates = doc_candidates[:k]
            else:
                # Fallback: just take top-k without reranking
                doc_candidates = doc_candidates[:k]
                if not EMBEDDINGS_FILTER_AVAILABLE:
                    logger.debug("[SEMANTIC_DOC_RETRIEVAL] EmbeddingsFilter not available, using top-k fallback")

            logger.info(f"[SEMANTIC_DOC_RETRIEVAL] Returning {len(doc_candidates)} docs for query: '{query[:50]}...'")

            return doc_candidates

        except Exception as e:
            import traceback

            logger.error(f"[SEMANTIC_DOC_RETRIEVAL] Error: {e}")
            logger.error(f"[SEMANTIC_DOC_RETRIEVAL] Traceback:\n{traceback.format_exc()}")
            return []
