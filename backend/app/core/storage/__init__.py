"""
Storage abstraction layer for Wikis.

Provides a backend-agnostic protocol for the unified wiki database
(graph nodes, edges, FTS, vector search, clustering, metadata).

Backends:
- ``sqlite`` — SQLite + FTS5 + sqlite-vec (default, zero-infra)
- ``postgres`` — PostgreSQL + tsvector/tsquery + pgvector (enterprise)

Usage::

    from app.core.storage import create_storage

    db = create_storage()          # reads WIKI_STORAGE_BACKEND from config
    db.from_networkx(graph)
    results = db.search_hybrid("authentication", embedding, k=20)
    db.close()
"""

from .factory import create_storage, drop_storage, is_postgres_backend, open_storage, repo_id_from_path
from .protocol import WikiStorageProtocol
from .text_index import StorageTextIndex

__all__ = [
    "StorageTextIndex",
    "WikiStorageProtocol",
    "create_storage",
    "drop_storage",
    "is_postgres_backend",
    "open_storage",
    "repo_id_from_path",
]
