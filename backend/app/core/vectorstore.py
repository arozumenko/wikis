"""
FAISS-based vector store manager with pluggable embeddings and code_graph retriever

An embeddings instance must be provided at construction; no fallback to HuggingFace.
"""

import copy
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Lazy docstore is enabled by default for memory efficiency
# Set WIKIS_DISABLE_LAZY_DOCSTORE=1 to use legacy pickle-based docstore
LAZY_DOCSTORE_ENABLED = os.getenv("WIKIS_DISABLE_LAZY_DOCSTORE") != "1"

# Feature flag for doc/code separation - affects cache key
# When enabled, vector store contains only docs (not code)
SEPARATE_DOC_INDEX = os.getenv("WIKIS_DOC_SEPARATE_INDEX", "0") == "1"


class DummyEmbeddings(Embeddings):
    """Dummy embeddings class for loading cached FAISS indexes"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dummy embeddings for documents"""
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return dummy embedding for query"""
        return [0.0] * 384


class VectorStoreManager:
    """Manager for FAISS vector stores with pluggable embeddings.

    An `embeddings` instance must be provided at construction.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        embeddings: Embeddings | None = None,
        embedding_batch_size: int | None = None,
    ):
        """
        Initialize VectorStoreManager.

        Args:
            cache_dir: Directory for caching vector stores
            embeddings: Pre-configured embeddings instance (required)
        """
        if embeddings is None:
            raise ValueError("embeddings must be provided to VectorStoreManager")

        if cache_dir is None:
            cache_dir = os.path.expanduser("./data/cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve batch size (env override -> argument -> default)
        env_batch = os.getenv("WIKI_EMBED_BATCH_SIZE")
        resolved_batch = None
        if embedding_batch_size is not None:
            resolved_batch = embedding_batch_size
        elif env_batch:
            try:
                resolved_batch = int(env_batch)
            except ValueError:
                logger.warning(f"Invalid WIKI_EMBED_BATCH_SIZE='{env_batch}', falling back to default 20")
        if not resolved_batch or resolved_batch <= 0:
            resolved_batch = 20
        self.embedding_batch_size = resolved_batch

        self.embeddings = embeddings
        try:
            emb_type = type(self.embeddings).__name__
            model_attr = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None)
            if model_attr:
                logger.info(f"Using injected embeddings: {emb_type} (model={model_attr})")
            else:
                logger.info(f"Using injected embeddings: {emb_type}")
        except Exception:
            logger.info("Using injected embeddings (details unavailable)")

        # Vector store and documents
        self.vectorstore: FAISS | None = None
        self.documents: list[Document] = []
        self.document_ids: dict[str, int] = {}  # UUID to index mapping
        self.cache_key: str | None = None

    def _maybe_copy_documents(self, documents: list[Document]) -> list[Document]:
        """Optionally deepcopy documents.

        By default, we avoid deepcopy to reduce memory usage.
        Set WIKIS_DEEPCOPY_DOCS=1 to enable deepcopy (legacy behavior).
        """
        if os.getenv("WIKIS_DEEPCOPY_DOCS") == "1":
            return copy.deepcopy(documents)
        return list(documents)

    def load_or_build(
        self, documents: list[Document], repo_path: str, force_rebuild: bool = False, commit_hash: str | None = None
    ) -> tuple[FAISS, list[Document]]:
        """
        Load existing vector store or build new one

        Args:
            documents: List of documents to index
            repo_path: Path to repository for cache key generation
            force_rebuild: Force rebuilding even if cache exists

        Returns:
            - FAISS vector store
            - List of indexed documents
        """
        # Don't mutate input list
        docs_copy = self._maybe_copy_documents(documents)

        # Generate cache key based on repository (path + commit hash / mtime)
        repo_hash = self._generate_repo_hash(repo_path, docs_copy, commit_hash)
        cache_file = self.cache_dir / f"{repo_hash}.faiss"
        docs_file = self.cache_dir / f"{repo_hash}.docs.pkl"

        # Try to load from cache
        if not force_rebuild and cache_file.exists() and docs_file.exists():
            try:
                logger.info(f"Loading vector store from cache: {cache_file}")
                return self._load_from_cache(cache_file, docs_file)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}, rebuilding...")

        # Build new vector store
        logger.info("Building new vector store...")
        return self._build_and_save(docs_copy, cache_file, docs_file)

    def load_or_build_from_iterable(
        self,
        documents_iter: Iterable[Document],
        repo_path: str,
        force_rebuild: bool = False,
        commit_hash: str | None = None,
    ) -> tuple[FAISS, list[Document]]:
        """Load vectorstore or build from a streaming iterable of documents."""
        repo_hash = self._generate_repo_hash(repo_path, [], commit_hash)
        cache_file = self.cache_dir / f"{repo_hash}.faiss"
        docs_file = self.cache_dir / f"{repo_hash}.docs.pkl"

        if not force_rebuild and cache_file.exists() and docs_file.exists():
            try:
                logger.info(f"Loading vector store from cache: {cache_file}")
                return self._load_from_cache(cache_file, docs_file)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}, rebuilding...")

        logger.info("Building new vector store from document stream...")
        return self._build_and_save_from_iterable(documents_iter, cache_file, docs_file)

    def _generate_repo_hash(self, repo_path: str, documents: list[Document], commit_hash: str | None) -> str:
        """Generate hash for repository content using (path + commit hash OR mtime) for stable cache identity.

        When WIKIS_DOC_SEPARATE_INDEX=1, includes a suffix to ensure separate cache
        for doc-only vector stores vs code+doc stores.
        """
        hasher = hashlib.md5()  # noqa: S324 — content fingerprint for cache key, not cryptographic use
        hasher.update(repo_path.encode())
        if commit_hash:
            hasher.update(commit_hash.encode())
        else:
            # Fallback to mtime if commit hash unavailable (non-git dirs)
            try:
                mtime = os.path.getmtime(repo_path)
                hasher.update(str(int(mtime)).encode())
            except OSError:
                pass

        base_hash = hasher.hexdigest()

        # When SEPARATE_DOC_INDEX is enabled, use different cache to avoid conflicts
        if SEPARATE_DOC_INDEX:
            return f"{base_hash}_doconly"
        return base_hash

    def _load_from_cache(self, cache_file: Path, docs_file: Path) -> tuple[FAISS, list[Document]]:
        """Load vector store and documents from cache"""
        cache_key = cache_file.stem
        self.cache_key = cache_key

        # Attempt lazy docstore load when enabled and available
        if LAZY_DOCSTORE_ENABLED:
            try:
                from .docstore import load_docstore_cache, migrate_docstore_from_docs_pickle

                lazy = load_docstore_cache(cache_file.parent, cache_key)
                if not lazy and docs_file.exists():
                    migrated = migrate_docstore_from_docs_pickle(cache_file.parent, cache_key, docs_file)
                    if migrated:
                        lazy = load_docstore_cache(cache_file.parent, cache_key)
                if lazy:
                    docstore, index_to_docstore_id, documents = lazy
                    faiss = dependable_faiss_import()
                    index = faiss.read_index(str(cache_file))
                    self.vectorstore = FAISS(
                        self.embeddings,
                        index,
                        docstore,
                        index_to_docstore_id,
                    )
                    self.documents = documents
                    # Load persisted mapping if available, otherwise rebuild
                    if not self._load_document_mapping(cache_file.parent, cache_key):
                        self._rebuild_document_mapping()
                    logger.info(f"Loaded {len(self.documents)} documents with lazy docstore from {cache_file}")
                    return self.vectorstore, self.documents
            except Exception as e:
                logger.warning(f"Lazy docstore load failed, falling back to pickle: {e}")

        # Load documents (legacy path)
        with open(docs_file, "rb") as f:
            self.documents = pickle.load(f)  # noqa: S301 — pickle used for FAISS docstore cache, data is self-generated

        # Load vector store
        self.vectorstore = FAISS.load_local(
            str(cache_file.parent),
            embeddings=self.embeddings,
            index_name=cache_key,
            allow_dangerous_deserialization=True,
        )

        # Load persisted document ID mapping if available, otherwise rebuild
        if not self._load_document_mapping(cache_file.parent, cache_key):
            self._rebuild_document_mapping()

        logger.info(f"Loaded {len(self.documents)} documents from cache")
        return self.vectorstore, self.documents

    def load_by_cache_key(self, cache_key: str) -> bool:
        """
        Load vector store by cache key (for Ask tool).

        This method allows loading an existing index without knowing the original
        documents or repository path. Used by the Ask tool which just needs to query.

        Args:
            cache_key: MD5 hash used as cache key (typically from repo path hash)

        Returns:
            True if loaded successfully, False if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.faiss"
        docs_file = self.cache_dir / f"{cache_key}.docs.pkl"

        # Check for lazy docstore files (new format) or legacy pickle
        docstore_bin = self.cache_dir / f"{cache_key}.docstore.bin"
        doc_index_json = self.cache_dir / f"{cache_key}.doc_index.json"
        has_lazy_docstore = docstore_bin.exists() and doc_index_json.exists()
        has_legacy_docs = docs_file.exists()

        if not cache_file.exists() or (not has_lazy_docstore and not has_legacy_docs):
            logger.debug(
                f"Cache not found for key {cache_key} (faiss={cache_file.exists()}, lazy={has_lazy_docstore}, legacy={has_legacy_docs})"
            )
            return False

        try:
            logger.info(
                f"Loading vectorstore from cache key={cache_key} "
                f"(faiss={cache_file.name}, lazy_docstore={has_lazy_docstore})"
            )
            self.cache_key = cache_key
            self._load_from_cache(cache_file, docs_file)
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache for key {cache_key}: {e}")
            return False

    def has_cache(self, cache_key: str) -> bool:
        """Check if cache exists for given key (supports lazy docstore or legacy pickle)"""
        cache_file = self.cache_dir / f"{cache_key}.faiss"
        docs_file = self.cache_dir / f"{cache_key}.docs.pkl"
        docstore_bin = self.cache_dir / f"{cache_key}.docstore.bin"
        doc_index_json = self.cache_dir / f"{cache_key}.doc_index.json"

        has_lazy_docstore = docstore_bin.exists() and doc_index_json.exists()
        has_legacy_docs = docs_file.exists()

        return cache_file.exists() and (has_lazy_docstore or has_legacy_docs)

    def _get_cache_index_path(self) -> Path:
        """Get path to cache index file that maps repo_identifier -> cache_key"""
        return self.cache_dir / "cache_index.json"

    def _load_cache_index(self) -> dict[str, Any]:
        """Load shared cache index.

        The index may include reserved top-level keys like 'graphs' and 'refs'.
        Vectorstore cache keys are stored at the top level keyed by repo_identifier.
        """
        index_path = self._get_cache_index_path()
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_cache_index(self, index: dict[str, Any]) -> None:
        """Save shared cache index (atomic write)."""
        index_path = self._get_cache_index_path()
        try:
            tmp_path = index_path.with_name(f"{index_path.name}.{uuid.uuid4().hex}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp_path, index_path)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def register_cache(self, repo_identifier: str, cache_key: str) -> None:
        """
        Register a cache key for a repo_identifier.
        Called after building vectorstore to enable lookup by repo name.
        """
        index = self._load_cache_index()
        index[repo_identifier] = cache_key

        # If repo_identifier is commit-scoped (repo:branch:commit8), store a canonical pointer.
        try:
            from .repo_resolution import ensure_refs, repo_branch_key, split_repo_identifier

            repo, branch, commit = split_repo_identifier(repo_identifier)
            if repo and branch and commit:
                refs = ensure_refs(index)
                refs[repo_branch_key(repo, branch)] = repo_identifier
        except Exception:  # noqa: S110
            # Never fail cache registration due to pointer bookkeeping.
            pass

        self._save_cache_index(index)
        logger.info(f"Registered cache: {repo_identifier} -> {cache_key}")

    def register_docstore_cache(self, repo_identifier: str, cache_key: str) -> None:
        """Register a docstore cache key for a repo_identifier."""
        index = self._load_cache_index()
        if "docs" not in index:
            index["docs"] = {}
        if isinstance(index["docs"], dict):
            index["docs"][repo_identifier] = cache_key

        # If repo_identifier is commit-scoped, store a canonical pointer for refs.
        try:
            from .repo_resolution import ensure_refs, repo_branch_key, split_repo_identifier

            repo, branch, commit = split_repo_identifier(repo_identifier)
            if repo and branch and commit:
                refs = ensure_refs(index)
                refs[repo_branch_key(repo, branch)] = repo_identifier
        except Exception:  # noqa: S110
            pass

        self._save_cache_index(index)
        logger.info(f"Registered docstore cache: {repo_identifier} -> {cache_key}")

    def load_by_repo_name(self, repo_identifier: str) -> FAISS | None:
        """
        Load vector store by repository identifier (convenience method for Ask tool).

        Looks up the cache key from the cache index file, then loads the cached index.
        Supports both old format (owner/repo:branch) and new format (owner/repo:branch:commit).

        Args:
            repo_identifier: Repository identifier (e.g., "owner/repo:branch" or "owner/repo:branch:commit")

        Returns:
            FAISS vector store if found, None otherwise
        """
        index = self._load_cache_index()

        # Resolve repo:branch -> repo:branch:commit8 deterministically.
        canonical_identifier = repo_identifier
        try:
            from .repo_resolution import resolve_canonical_repo_identifier

            canonical_identifier = resolve_canonical_repo_identifier(
                repo_identifier=repo_identifier,
                cache_dir=self.cache_dir,
                repositories_dir=self.cache_dir / "repositories",
            )
        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception:
            canonical_identifier = repo_identifier

        cache_key = index.get(canonical_identifier)
        if cache_key is None and canonical_identifier != repo_identifier:
            cache_key = index.get(repo_identifier)

        if not cache_key:
            logger.warning(f"No cache index entry for {repo_identifier}")
            # Fallback: try to find any available cache (for single-repo setups)
            available_caches = list(self.cache_dir.glob("*.faiss"))
            if len(available_caches) == 1:
                cache_key = available_caches[0].stem
                logger.info(f"Using single available cache: {cache_key}")
            else:
                return None

        if self.load_by_cache_key(cache_key):
            return self.vectorstore
        return None

    def _build_and_save(
        self, documents: list[Document], cache_file: Path, docs_file: Path
    ) -> tuple[FAISS, list[Document]]:
        """Build vector store and save to cache"""
        if not documents:
            raise ValueError("No documents to index")
        self.cache_key = cache_file.stem

        # Assign UUIDs to documents if not present
        for doc in documents:
            if "uuid" not in doc.metadata:
                doc.metadata["uuid"] = str(uuid.uuid4())

        # Filter out documents with empty content
        # AWS Titan and other embedding models reject empty strings (minLength: 1)
        # Also normalize Unicode whitespace (NBSP, Zero-Width Space, etc.) that .strip() misses
        valid_documents = []
        empty_count = 0

        for doc in documents:
            content = (doc.page_content or "").strip()

            # Replace ALL Unicode whitespace/control chars with regular space
            # This catches NBSP (U+00A0), Zero-Width Space (U+200B), etc.
            # Preserves word boundaries while normalizing invisible characters
            import unicodedata

            normalized = "".join(
                " " if unicodedata.category(char).startswith(("Z", "C")) else char for char in content
            ).strip()

            if normalized:  # Non-empty after normalizing Unicode whitespace
                # Update document content with normalized version
                doc.page_content = normalized
                valid_documents.append(doc)
            else:
                empty_count += 1
                # Log the actual content bytes for debugging Unicode issues
                content_repr = repr(content[:100]) if content else "''"
                logger.warning(
                    f"Skipping document with empty/whitespace-only content: "
                    f"source={doc.metadata.get('source', 'unknown')}, "
                    f"chunk_type={doc.metadata.get('chunk_type', 'unknown')}, "
                    f"symbol_name={doc.metadata.get('symbol_name', 'N/A')}, "
                    f"content_preview={content_repr}"
                )

        if empty_count > 0:
            logger.warning(f"Filtered out {empty_count} empty documents (total attempted: {len(documents)})")

        if not valid_documents:
            raise ValueError(
                f"No valid documents to index - all {len(documents)} documents had empty content. "
                f"This may indicate parser issues or binary files being processed."
            )

        self.documents = valid_documents

        # Extract text content for embedding, truncating to avoid token limits
        # (e.g., AWS Titan Embed has 8192 token / ~30K char limit)
        max_embed_chars = int(os.getenv("WIKI_MAX_EMBED_CHARS", "30000"))
        texts = []
        truncated_count = 0
        for doc in self.documents:
            text = doc.page_content
            if len(text) > max_embed_chars:
                text = text[:max_embed_chars]
                truncated_count += 1
            texts.append(text)
        if truncated_count:
            logger.warning(f"Truncated {truncated_count} documents to {max_embed_chars} chars for embedding")
        metadatas = [doc.metadata for doc in self.documents]

        total_docs = len(texts)
        batch_size = self.embedding_batch_size
        logger.info(f"Embedding {total_docs} documents (batch_size={batch_size})...")
        start_time = time.time()

        if total_docs <= batch_size:
            # Single-shot embedding for small document sets
            try:
                self.vectorstore = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)
            except Exception as e:
                logger.error(f"Failed to embed documents: {e}")
                # Log sample of problematic documents for debugging
                for i, (text, meta) in enumerate(list(zip(texts, metadatas, strict=False))[:5]):
                    logger.debug(
                        f"  Sample doc {i}: content_len={len(text)}, "
                        f"source={meta.get('source', 'unknown')}, "
                        f"chunk_type={meta.get('chunk_type', 'unknown')}"
                    )
                raise
        else:
            total_batches = (total_docs + batch_size - 1) // batch_size
            self.vectorstore = None
            failed_docs_total = 0
            successful_docs = 0

            for batch_index, start in enumerate(range(0, total_docs, batch_size), start=1):
                end = start + batch_size
                batch_texts = texts[start:end]
                batch_metas = metadatas[start:end]
                bcount = len(batch_texts)

                try:
                    if self.vectorstore is None:
                        self.vectorstore = FAISS.from_texts(
                            texts=batch_texts, embedding=self.embeddings, metadatas=batch_metas
                        )
                    else:
                        self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)
                    successful_docs += bcount
                    logger.info(
                        f"Embedded batch {batch_index}/{total_batches} "
                        f"(docs {start + 1}-{start + bcount}/{total_docs}, batch_size={bcount})"
                    )

                except Exception as e:
                    logger.warning(
                        f"Batch {batch_index}/{total_batches} failed: {e}. "
                        f"Attempting per-document embedding to isolate failures..."
                    )

                    # If this is the first batch, try one-by-one to initialize vectorstore
                    # Otherwise, add documents one-by-one to existing vectorstore
                    batch_failed_docs = 0
                    batch_success_docs = 0

                    for doc_idx, (text, meta) in enumerate(zip(batch_texts, batch_metas, strict=True)):
                        try:
                            # Normalize Unicode whitespace that may have slipped through
                            # Replace NBSP, Zero-Width Space, etc. with regular space
                            import unicodedata

                            normalized = "".join(
                                " " if unicodedata.category(char).startswith(("Z", "C")) else char for char in text
                            ).strip()

                            if not normalized:
                                # Skip documents with only Unicode whitespace/control chars
                                batch_failed_docs += 1
                                content_repr = repr(text[:100])
                                logger.warning(
                                    f"  Skipping document {doc_idx + 1}/{bcount} in batch {batch_index} "
                                    f"(Unicode whitespace only): content_preview={content_repr}, "
                                    f"source={meta.get('source', 'unknown')}"
                                )
                                continue

                            # Use normalized text for embedding
                            if self.vectorstore is None:
                                # First successful document initializes the vectorstore
                                self.vectorstore = FAISS.from_texts(
                                    texts=[normalized], embedding=self.embeddings, metadatas=[meta]
                                )
                            else:
                                self.vectorstore.add_texts(texts=[normalized], metadatas=[meta])
                            batch_success_docs += 1

                        except Exception as doc_error:
                            batch_failed_docs += 1
                            logger.error(
                                f"  Failed to embed document {doc_idx + 1}/{bcount} in batch {batch_index}: "
                                f"content_len={len(text)}, "
                                f"source={meta.get('source', 'unknown')}, "
                                f"chunk_type={meta.get('chunk_type', 'unknown')}, "
                                f"error={str(doc_error)[:100]}"
                            )

                    successful_docs += batch_success_docs
                    failed_docs_total += batch_failed_docs

                    logger.info(
                        f"Batch {batch_index}/{total_batches} completed with per-document fallback: "
                        f"{batch_success_docs} succeeded, {batch_failed_docs} failed"
                    )

                    # If no vectorstore initialized yet (all docs in first batch failed)
                    if self.vectorstore is None:
                        logger.error("All documents in first batch failed - cannot initialize vectorstore")
                        raise ValueError(
                            f"Failed to initialize vectorstore: all {bcount} documents in first batch "
                            f"failed embedding. This may indicate a systematic issue with the embedding model."
                        ) from e

            elapsed = time.time() - start_time

            if failed_docs_total > 0:
                logger.warning(
                    f"Completed embedding with {failed_docs_total} failed documents: "
                    f"{successful_docs}/{total_docs} documents successfully indexed "
                    f"(elapsed={elapsed:.2f}s)"
                )
            else:
                logger.info(
                    f"Completed embedding of {total_docs} documents in {total_batches} batches (elapsed={elapsed:.2f}s)"
                )

        # Build document ID mapping
        self._rebuild_document_mapping()

        # Save to cache
        try:
            logger.info(f"Saving vector store to cache: {cache_file}")
            self.vectorstore.save_local(str(cache_file.parent), index_name=cache_file.stem)

            # Save documents
            with open(docs_file, "wb") as f:
                pickle.dump(self.documents, f)

            # Save document ID mapping for fast loading (avoids O(n) rebuild on cache hit)
            self._save_document_mapping(cache_file.parent, cache_file.stem)

            # Save mmap-friendly docstore cache for lazy loading
            try:
                from .docstore import build_docstore_cache

                build_docstore_cache(self.documents, cache_file.parent, cache_file.stem)
            except Exception as e:
                logger.warning(f"Failed to save lazy docstore cache: {e}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

        logger.info(f"Built vector store with {len(self.documents)} documents")
        if LAZY_DOCSTORE_ENABLED and os.getenv("WIKIS_RELOAD_LAZY_DOCSTORE_AFTER_BUILD", "1") == "1":
            try:
                logger.info("Reloading vector store with lazy docstore after build")
                return self._load_from_cache(cache_file, docs_file)
            except Exception as e:
                logger.warning(f"Failed to reload lazy docstore after build: {e}")

        return self.vectorstore, self.documents

    def _build_and_save_from_iterable(
        self,
        documents_iter: Iterable[Document],
        cache_file: Path,
        docs_file: Path,
    ) -> tuple[FAISS, list[Document]]:
        """Build vector store from a streaming iterable to avoid RAM spikes."""
        self.cache_key = cache_file.stem
        import numpy as np

        faiss = dependable_faiss_import()
        batch_size = self.embedding_batch_size

        docstore_path = cache_file.parent / f"{cache_file.stem}.docstore.bin"
        entries: dict[str, dict[str, Any]] = {}
        doc_ids: list[str] = []
        index_to_docstore_id: dict[int, str] = {}
        documents_meta: list[Document] = []

        index = None
        total_docs = 0
        skipped_docs = 0
        current_index = 0

        def _normalize_content(text: str) -> str:
            import unicodedata

            normalized = "".join(
                " " if unicodedata.category(char).startswith(("Z", "C")) else char for char in text
            ).strip()
            return normalized

        with open(docstore_path, "wb") as docstore_file:
            batch_texts: list[str] = []
            batch_doc_ids: list[str] = []
            batch_index = 0
            start_time = time.time()

            for doc in documents_iter:
                content = _normalize_content(doc.page_content or "")
                if not content:
                    source = doc.metadata.get("source") or doc.metadata.get("rel_path") or doc.metadata.get("file_path")
                    symbol_name = doc.metadata.get("symbol_name") or doc.metadata.get("symbol")
                    chunk_type = doc.metadata.get("chunk_type")
                    raw_preview = (doc.page_content or "")[:120]
                    logger.warning(
                        "Skipping document with empty content after normalization: "
                        f"source={source}, chunk_type={chunk_type}, symbol={symbol_name}, "
                        f"raw_preview={raw_preview!r}"
                    )
                    skipped_docs += 1
                    continue

                doc_id = doc.metadata.get("uuid")
                if not doc_id:
                    doc_id = str(uuid.uuid4())
                    doc.metadata["uuid"] = doc_id

                encoded = content.encode("utf-8")
                offset = docstore_file.tell()
                docstore_file.write(encoded)
                length = len(encoded)

                entries[str(doc_id)] = {
                    "offset": offset,
                    "length": length,
                    "metadata": doc.metadata,
                }
                doc_ids.append(str(doc_id))

                batch_texts.append(content)
                batch_doc_ids.append(str(doc_id))
                documents_meta.append(Document(page_content="", metadata=doc.metadata))

                if len(batch_texts) >= batch_size:
                    batch_index += 1
                    embeddings = self.embeddings.embed_documents(batch_texts)
                    embeddings_np = np.asarray(embeddings, dtype="float32")

                    if index is None:
                        index = faiss.IndexFlatL2(len(embeddings_np[0]))

                    index.add(embeddings_np)

                    for doc_id_value in batch_doc_ids:
                        index_to_docstore_id[current_index] = doc_id_value
                        current_index += 1

                    total_docs += len(batch_texts)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Embedded batch {batch_index} "
                        f"(docs {total_docs - len(batch_texts) + 1}-{total_docs}, "
                        f"batch_size={len(batch_texts)}, elapsed={elapsed:.2f}s)"
                    )
                    batch_texts = []
                    batch_doc_ids = []

            if batch_texts:
                batch_index += 1
                embeddings = self.embeddings.embed_documents(batch_texts)
                embeddings_np = np.asarray(embeddings, dtype="float32")
                if index is None:
                    index = faiss.IndexFlatL2(len(embeddings_np[0]))
                index.add(embeddings_np)
                for doc_id_value in batch_doc_ids:
                    index_to_docstore_id[current_index] = doc_id_value
                    current_index += 1
                total_docs += len(batch_texts)
                elapsed = time.time() - start_time
                logger.info(
                    f"Embedded batch {batch_index} "
                    f"(docs {total_docs - len(batch_texts) + 1}-{total_docs}, "
                    f"batch_size={len(batch_texts)}, elapsed={elapsed:.2f}s)"
                )

        if index is None:
            raise ValueError("No valid documents to index from stream")

        try:
            logger.info(f"Saving vector store to cache: {cache_file}")
            faiss.write_index(index, str(cache_file))

            # Save docstore index for lazy loading
            try:
                from .docstore import write_docstore_index

                write_docstore_index(cache_file.parent, cache_file.stem, docstore_path, entries, doc_ids)
            except Exception as e:
                logger.warning(f"Failed to save lazy docstore index: {e}")

            # Save documents pickle only when lazy loading is disabled
            if not LAZY_DOCSTORE_ENABLED:
                with open(docs_file, "wb") as f:
                    pickle.dump(documents_meta, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

        try:
            from .docstore import MMapDocstore

            docstore = MMapDocstore(docstore_path, entries)
        except Exception:
            from langchain_community.docstore.in_memory import InMemoryDocstore

            docstore = InMemoryDocstore({})

        self.vectorstore = FAISS(
            self.embeddings,
            index,
            docstore,
            index_to_docstore_id,
        )
        self.documents = documents_meta
        self._rebuild_document_mapping()

        logger.info(f"Built vector store from stream with {total_docs} documents (skipped={skipped_docs})")

        return self.vectorstore, self.documents

    def _rebuild_document_mapping(self):
        """Rebuild the UUID to index mapping"""
        self.document_ids = {}
        for i, doc in enumerate(self.documents):
            if "uuid" in doc.metadata:
                self.document_ids[doc.metadata["uuid"]] = i

    def _save_document_mapping(self, cache_dir: Path, cache_key: str):
        """Save UUID to index mapping to disk for fast loading."""
        mapping_file = cache_dir / f"{cache_key}.docids.json"
        try:
            import json

            with open(mapping_file, "w") as f:
                json.dump(self.document_ids, f)
            logger.debug(f"Saved document ID mapping to {mapping_file}")
        except Exception as e:
            logger.warning(f"Failed to save document ID mapping: {e}")

    def _load_document_mapping(self, cache_dir: Path, cache_key: str) -> bool:
        """Load UUID to index mapping from disk. Returns True if successful."""
        mapping_file = cache_dir / f"{cache_key}.docids.json"
        if not mapping_file.exists():
            return False
        try:
            import json

            with open(mapping_file) as f:
                self.document_ids = json.load(f)
            # Convert string keys back to int values (JSON serializes as strings)
            self.document_ids = {k: int(v) for k, v in self.document_ids.items()}
            logger.debug(f"Loaded document ID mapping from {mapping_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load document ID mapping: {e}")
            return False

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add new documents to the vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        # Don't mutate input list
        docs_copy = self._maybe_copy_documents(documents)

        # Assign UUIDs if not present
        doc_ids = []
        for doc in docs_copy:
            if "uuid" not in doc.metadata:
                doc_id = str(uuid.uuid4())
                doc.metadata["uuid"] = doc_id
            else:
                doc_id = doc.metadata["uuid"]
            doc_ids.append(doc_id)

        # Extract texts and metadatas
        texts = [doc.page_content for doc in docs_copy]
        metadatas = [doc.metadata for doc in docs_copy]

        # Add to FAISS
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

        # Update document lists and mapping
        start_idx = len(self.documents)
        self.documents.extend(docs_copy)

        for i, doc_id in enumerate(doc_ids):
            self.document_ids[doc_id] = start_idx + i

        return doc_ids

    def delete_documents(self, document_ids: list[str]):
        """Delete documents by UUID using efficient FAISS deletion"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        # Find docstore IDs to delete
        docstore_ids_to_delete = []
        indices_to_remove = []

        for i, doc in enumerate(self.documents):
            doc_uuid = doc.metadata.get("uuid")
            if doc_uuid in document_ids:
                # Find the docstore ID for this document
                # Index in self.documents corresponds to index in vectorstore
                if i in self.vectorstore.index_to_docstore_id:
                    docstore_id = self.vectorstore.index_to_docstore_id[i]
                    docstore_ids_to_delete.append(docstore_id)
                    indices_to_remove.append(i)

        if not docstore_ids_to_delete:
            # No documents to delete
            return

        # Delete from FAISS vectorstore (efficient, no rebuilding)
        self.vectorstore.delete(docstore_ids_to_delete)

        # Remove documents from our list (in reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]

        # Rebuild document mapping to match new vectorstore state
        self._rebuild_document_mapping()

    def search(self, query: str, k: int = 10, filter_dict: dict | None = None) -> list[Document]:
        """Search documents with optional filtering"""
        if not self.vectorstore:
            return []

        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self.vectorstore.similarity_search(query, k=k)

    def search_with_score(
        self, query: str, k: int = 10, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Search with similarity scores"""
        if not self.vectorstore:
            return []

        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Convert numpy.float32 to float for JSON serialization
        return [(doc, float(score)) for doc, score in results]

    def search_by_type(self, query: str, chunk_type: str, k: int = 10) -> list[Document]:
        """Search documents by chunk type (code, text, etc.)"""
        filter_dict = {"chunk_type": {"$eq": chunk_type}}
        return self.search(query, k=k, filter_dict=filter_dict)

    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict | None = None):
        """Convert to LangChain retriever"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        if search_kwargs is None:
            search_kwargs = {"k": 10}

        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def get_vectorstore(self) -> FAISS | None:
        """Get the FAISS vector store"""
        return self.vectorstore

    def get_all_documents(self) -> list[Document]:
        """Get all indexed documents"""
        return self.documents.copy()  # Return copy to prevent mutation

    def get_mmap_bm25_retriever(
        self,
        k: int = 25,
        rebuild: bool = False,
    ):
        """Get disk-backed BM25 retriever using mmap docstore when available."""
        cache_key = self.cache_key
        if not cache_key:
            return None

        use_mmap = os.getenv("WIKIS_MMAP_BM25", "1")
        if use_mmap == "0":
            return None

        try:
            from .bm25_disk import MMapBM25Retriever

            retriever = MMapBM25Retriever.from_cache(
                cache_dir=self.cache_dir,
                cache_key=cache_key,
                k=k,
                rebuild=rebuild,
            )
            return retriever
        except Exception as e:
            logger.warning(f"Failed to initialize mmap BM25 retriever: {e}")
            return None

    def clear_cache(self, repo_path: str | None = None, commit_hash: str | None = None):
        """Clear cached vector stores"""
        if repo_path:
            # Clear specific repository cache
            docs = []  # Empty for hash calculation
            repo_hash = self._generate_repo_hash(repo_path, docs, commit_hash)
            cache_file = self.cache_dir / f"{repo_hash}.faiss"
            docs_file = self.cache_dir / f"{repo_hash}.docs.pkl"

            for file_path in [cache_file, docs_file]:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed cache file: {file_path}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.faiss"):
                cache_file.unlink()
            for docs_file in self.cache_dir.glob("*.docs.pkl"):
                docs_file.unlink()
            logger.info("Cleared all vector store cache")
