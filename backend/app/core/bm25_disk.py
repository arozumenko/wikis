"""Disk-backed BM25 index + retriever backed by mmap docstore."""

from __future__ import annotations

import logging
import math
import mmap
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .docstore import DocstoreIndex, MMapDocstore, migrate_docstore_from_docs_pickle

logger = logging.getLogger(__name__)

BM25_SQLITE_SCHEMA_VERSION = 1
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

try:
    from langchain_community.retrievers.bm25 import default_preprocessing_func
except Exception:
    default_preprocessing_func = None


def _default_tokenizer(text: str) -> list[str]:
    if default_preprocessing_func:
        try:
            tokens = default_preprocessing_func(text)
        except Exception:
            tokens = None
        if isinstance(tokens, list):
            return [t for t in tokens if t]
    return [t for t in text.split() if t]


class BM25SqliteIndex:
    """Disk-backed BM25 postings index stored in SQLite.

    Thread/greenlet safety: Each search() call opens its own read-only connection.
    This prevents cursor corruption when multiple gevent greenlets call search()
    concurrently on the same index instance (e.g. parallel tool calls in DeepAgents).
    """

    def __init__(
        self,
        db_path: Path,
        doc_ids: list[str],
        doc_lengths: list[int],
        avgdl: float,
        k1: float,
        b: float,
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        self._db_path = db_path
        self._doc_ids = doc_ids
        self._doc_lengths = doc_lengths
        self._avgdl = avgdl if avgdl > 0 else 1.0
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer

    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)

    def _open_conn(self) -> sqlite3.Connection:
        """Open a fresh read-only SQLite connection.

        Each search() call gets its own connection so parallel gevent greenlets
        never share cursors.  Read-only SQLite connections are very cheap.
        """
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.execute("PRAGMA query_only = ON")
        return conn

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search the BM25 index for relevant documents.

        Opens a dedicated read-only connection for this call so that parallel
        searches under gevent do not interfere with each other.

        Returns empty list on transient errors rather than crashing.
        """
        tokens = self._tokenizer(query or "")
        if not tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        doc_count = self.doc_count
        if doc_count == 0:
            return []

        doc_lengths_count = len(self._doc_lengths)
        conn: sqlite3.Connection | None = None

        try:
            conn = self._open_conn()

            term_counts = Counter(tokens)
            for term, query_tf in term_counts.items():
                df_row = conn.execute(
                    "SELECT df FROM terms WHERE term = ?",
                    (term,),
                ).fetchone()
                if not df_row:
                    continue
                df = int(df_row[0])
                if df <= 0:
                    continue

                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))

                for doc_idx, tf in conn.execute(
                    "SELECT doc_idx, tf FROM postings WHERE term = ?",
                    (term,),
                ):
                    # Guard against NULL or out-of-range (corrupted / legacy index)
                    if doc_idx is None or tf is None:
                        continue
                    if doc_idx < 0 or doc_idx >= doc_lengths_count:
                        continue

                    dl = self._doc_lengths[doc_idx] or 1
                    denom = tf + self._k1 * (1 - self._b + self._b * (dl / self._avgdl))
                    if denom <= 0:
                        continue
                    score = idf * ((tf * (self._k1 + 1)) / denom)
                    scores[doc_idx] += score * query_tf

        except (sqlite3.InterfaceError, sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.warning(f"BM25 search failed: {e}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

        if not scores:
            return []

        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        results: list[tuple[str, float]] = []
        for doc_idx, score in top:
            if doc_idx < len(self._doc_ids):
                results.append((self._doc_ids[doc_idx], score))
        return results

    @classmethod
    def load(
        cls,
        db_path: Path,
        tokenizer: Callable[[str], list[str]],
    ) -> BM25SqliteIndex | None:
        if not db_path.exists():
            return None

        conn: sqlite3.Connection | None = None
        try:
            # Open a temporary connection just to read metadata and doc lists.
            # search() will open its own per-call connections.
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.execute("PRAGMA query_only = ON")
            conn.row_factory = sqlite3.Row

            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            avgdl = float(meta.get("avgdl", 1.0))
            k1 = float(meta.get("k1", DEFAULT_BM25_K1))
            b = float(meta.get("b", DEFAULT_BM25_B))

            rows = conn.execute("SELECT doc_idx, doc_id, length FROM docs ORDER BY doc_idx").fetchall()
            doc_ids = [row["doc_id"] for row in rows]
            doc_lengths = [int(row["length"]) for row in rows]

            conn.close()
            conn = None

            return cls(
                db_path=Path(db_path),
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                k1=k1,
                b=b,
                tokenizer=tokenizer,
            )
        except Exception as e:
            logger.warning(f"Failed to load BM25 index {db_path}: {e}")
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE docs (doc_idx INTEGER PRIMARY KEY, doc_id TEXT, length INTEGER)")
    conn.execute("CREATE TABLE postings (term TEXT, doc_idx INTEGER, tf INTEGER)")
    conn.execute("CREATE TABLE terms (term TEXT PRIMARY KEY, df INTEGER)")


def build_bm25_index(
    cache_dir: Path,
    cache_key: str,
    tokenizer: Callable[[str], list[str]] = _default_tokenizer,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
    rebuild: bool = False,
) -> Path | None:
    """Build BM25 SQLite index from docstore cache (streaming, low memory)."""
    cache_dir = Path(cache_dir)
    db_path = cache_dir / f"{cache_key}.bm25.sqlite"

    if db_path.exists() and not rebuild:
        return db_path

    docstore_index = DocstoreIndex.load(cache_dir, cache_key)
    if not docstore_index:
        docs_file = cache_dir / f"{cache_key}.docs.pkl"
        migrated = migrate_docstore_from_docs_pickle(cache_dir, cache_key, docs_file)
        if migrated:
            docstore_index = DocstoreIndex.load(cache_dir, cache_key)
    if not docstore_index:
        logger.warning(f"BM25 build skipped: docstore index missing for {cache_key}")
        return None

    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:  # noqa: S110
            pass

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        _initialize_schema(conn)

        total_len = 0
        doc_count = 0
        commit_every = 200
        skipped_count = 0

        with open(docstore_index.docstore_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                conn.execute("BEGIN")
                # Use sequential doc_idx counter instead of enumerate() to avoid gaps
                # When entries are skipped, enumerate() creates gaps (0,1,3,4) but
                # the loading code expects contiguous indices (0,1,2,3)
                sequential_doc_idx = 0

                for doc_id in docstore_index.doc_ids:
                    entry = docstore_index.entries.get(doc_id)
                    if not entry:
                        skipped_count += 1
                        continue

                    offset = int(entry.get("offset", 0))
                    length = int(entry.get("length", 0))
                    if length <= 0:
                        content = ""
                    else:
                        content = mm[offset : offset + length].decode("utf-8", errors="replace")

                    tokens = tokenizer(content)
                    doc_length = len(tokens)

                    # Skip documents with no tokens (empty content after tokenization)
                    if doc_length == 0:
                        skipped_count += 1
                        continue

                    total_len += doc_length
                    doc_count += 1

                    conn.execute(
                        "INSERT INTO docs (doc_idx, doc_id, length) VALUES (?, ?, ?)",
                        (sequential_doc_idx, str(doc_id), doc_length),
                    )

                    term_freq = Counter(tokens)
                    postings = [(term, sequential_doc_idx, int(tf)) for term, tf in term_freq.items()]
                    conn.executemany(
                        "INSERT INTO postings (term, doc_idx, tf) VALUES (?, ?, ?)",
                        postings,
                    )

                    # Increment sequential counter only after successful insert
                    sequential_doc_idx += 1

                    if doc_count % commit_every == 0:
                        conn.commit()
                        conn.execute("BEGIN")

                conn.commit()
            finally:
                mm.close()

        if skipped_count > 0:
            logger.info(f"BM25 build: skipped {skipped_count} documents (missing entry or empty content)")

        conn.execute("INSERT INTO terms (term, df) SELECT term, COUNT(*) FROM postings GROUP BY term")
        conn.execute("CREATE INDEX idx_postings_term ON postings(term)")
        conn.execute("CREATE INDEX idx_docs_doc_idx ON docs(doc_idx)")

        avgdl = float(total_len) / doc_count if doc_count else 1.0
        if avgdl <= 0:
            avgdl = 1.0
        meta_rows = [
            ("schema_version", str(BM25_SQLITE_SCHEMA_VERSION)),
            ("doc_count", str(doc_count)),
            ("avgdl", str(avgdl)),
            ("k1", str(k1)),
            ("b", str(b)),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)
        conn.commit()
        conn.close()

        _register_bm25_cache(cache_dir, cache_key)

        logger.info(f"Built BM25 index: {db_path.name} (docs={doc_count})")
        return db_path
    except Exception as e:
        logger.warning(f"Failed to build BM25 index for {cache_key}: {e}")
        return None


def load_or_build_bm25_index(
    cache_dir: Path,
    cache_key: str,
    tokenizer: Callable[[str], list[str]] = _default_tokenizer,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
    rebuild: bool = False,
) -> BM25SqliteIndex | None:
    db_path = cache_dir / f"{cache_key}.bm25.sqlite"
    index = BM25SqliteIndex.load(db_path, tokenizer=tokenizer)
    if index and not rebuild:
        return index

    built = build_bm25_index(
        cache_dir=cache_dir,
        cache_key=cache_key,
        tokenizer=tokenizer,
        k1=k1,
        b=b,
        rebuild=rebuild,
    )
    if not built:
        return None

    return BM25SqliteIndex.load(db_path, tokenizer=tokenizer)


def _register_bm25_cache(cache_dir: Path, cache_key: str) -> None:
    """Register bm25 cache key in cache_index.json if repo_identifier can be inferred."""
    try:
        from .repo_resolution import load_cache_index, save_cache_index_atomic

        index = load_cache_index(cache_dir)
        if not isinstance(index, dict):
            return

        bm25_map = index.get("bm25")
        if not isinstance(bm25_map, dict):
            bm25_map = {}
            index["bm25"] = bm25_map

        repo_identifier = None
        for key, value in index.items():
            if key in {"graphs", "refs", "docs", "bm25"}:
                continue
            if value == cache_key:
                repo_identifier = key
                break

        if repo_identifier:
            bm25_map[repo_identifier] = cache_key
            save_cache_index_atomic(cache_dir, index)
    except Exception as e:
        logger.warning(f"Failed to register bm25 cache: {e}")


class MMapBM25Retriever(BaseRetriever):
    """BM25 retriever backed by mmap docstore + disk index."""

    index: BM25SqliteIndex
    docstore: MMapDocstore
    k: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path,
        cache_key: str,
        k: int = 10,
        rebuild: bool = False,
        tokenizer: Callable[[str], list[str]] = _default_tokenizer,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> MMapBM25Retriever | None:
        index = load_or_build_bm25_index(
            cache_dir=Path(cache_dir),
            cache_key=cache_key,
            tokenizer=tokenizer,
            k1=k1,
            b=b,
            rebuild=rebuild,
        )
        if not index:
            return None

        docstore_index = DocstoreIndex.load(Path(cache_dir), cache_key)
        if not docstore_index:
            return None

        docstore = MMapDocstore(docstore_index.docstore_path, docstore_index.entries)
        return cls(index=index, docstore=docstore, k=k)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        hits = self.index.search(query, self.k)
        documents: list[Document] = []
        for doc_id, _score in hits:
            doc = self.docstore.search(doc_id)
            if isinstance(doc, Document):
                documents.append(doc)
        return documents
