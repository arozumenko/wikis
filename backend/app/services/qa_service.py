"""Q&A Knowledge Flywheel service — cache lookup, recording, listing, stats, invalidation."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import case, delete as sa_delete, func, select, update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.models.db_models import QARecord, WikiRecord
from app.services.qa_cache_manager import QACacheManager

logger = logging.getLogger(__name__)


def _compute_question_hash(question: str) -> str:
    """SHA-256 hex digest of the question text. Single source of truth for hash computation."""
    return hashlib.sha256(question.encode()).hexdigest()


class QAService:
    """Orchestrates QA knowledge flywheel operations."""

    def __init__(
        self,
        session_factory: async_sessionmaker,
        cache: QACacheManager | None,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._cache = cache
        self._settings = settings

    async def get_wiki_commit_hash(self, wiki_id: str) -> str | None:
        """Get the current commit hash for a wiki."""
        async with self._session_factory() as session:
            result = await session.execute(select(WikiRecord.commit_hash).where(WikiRecord.id == wiki_id))
            row = result.scalar_one_or_none()
            return row

    async def _rebuild_cache_from_db(self, wiki_id: str) -> None:
        """Fetch cache-eligible records from DB and rebuild the FAISS index.

        Applies the same commit + TTL filters as lookup_cache to prevent
        expired or old-commit rows from dominating the rebuilt index.
        """
        if self._cache is None:
            return
        current_commit = await self.get_wiki_commit_hash(wiki_id)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._settings.qa_cache_max_age_seconds)

        async with self._session_factory() as session:
            stmt = select(QARecord).where(
                QARecord.wiki_id == wiki_id,
                QARecord.status.in_(["pending", "validated", "enriched"]),
                QARecord.has_context == 0,
                QARecord.created_at > cutoff.replace(tzinfo=None),
            )
            if current_commit:
                stmt = stmt.where(QARecord.source_commit_hash == current_commit)
            result = await session.execute(stmt)
            eligible = list(result.scalars().all())
        try:
            await self._cache.rebuild_from_records(wiki_id, eligible)
            logger.info(
                "QA index rebuilt for wiki %s from %d eligible records",
                wiki_id,
                len(eligible),
            )
        except Exception:
            logger.error("Failed to rebuild QA index for wiki %s", wiki_id, exc_info=True)

    async def lookup_cache(
        self,
        wiki_id: str,
        question: str,
        chat_history: list | None = None,
    ) -> tuple[QARecord | None, np.ndarray | None]:
        """Two-layer cache lookup. Returns (cached_record, reusable_embedding)."""
        # Disabled
        if not self._settings.qa_cache_enabled:
            return (None, None)

        # Context bypass — conversational follow-ups are not cacheable
        if chat_history:
            return (None, None)

        # No cache manager — recording-only mode
        if self._cache is None:
            return (None, None)

        current_commit = await self.get_wiki_commit_hash(wiki_id)
        question_hash = _compute_question_hash(question)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._settings.qa_cache_max_age_seconds)

        # Layer 1: Exact hash match (DB)
        async with self._session_factory() as session:
            stmt = (
                select(QARecord)
                .where(
                    QARecord.wiki_id == wiki_id,
                    QARecord.question_hash == question_hash,
                    QARecord.status.in_(["pending", "validated", "enriched"]),
                    QARecord.has_context == 0,
                )
                .order_by(QARecord.created_at.desc())
                .limit(1)
            )
            # Commit-awareness filter — NULL commit hashes are excluded when
            # current_commit is known (they may come from the refresh window)
            if current_commit:
                stmt = stmt.where(QARecord.source_commit_hash == current_commit)
            # TTL filter — compare against timezone-naive cutoff for SQLite compat
            stmt = stmt.where(QARecord.created_at > cutoff.replace(tzinfo=None))

            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            if record:
                return (record, None)  # No embedding computed

        # Layer 2: FAISS semantic search (k=5)
        # Before searching, check if the index is corrupt and needs rebuild
        if self._cache.check_needs_rebuild(wiki_id):
            logger.info("QA index for wiki %s needs rebuild — recovering from DB", wiki_id)
            await self._rebuild_cache_from_db(wiki_id)

        qa_ids, embedding = await self._cache.search(
            wiki_id,
            question,
            self._settings.qa_cache_similarity_threshold,
            k=5,
        )

        if qa_ids:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(QARecord).where(QARecord.id.in_(qa_ids))
                )
                records_by_id = {r.id: r for r in result.scalars().all()}

                # Walk in ranked order (best match first) and apply eligibility filters
                cutoff_naive = cutoff.replace(tzinfo=None)
                for qa_id in qa_ids:
                    record = records_by_id.get(qa_id)
                    if record is None:
                        continue
                    if record.status not in ("pending", "validated", "enriched"):
                        continue
                    if record.has_context:
                        continue
                    if current_commit and record.source_commit_hash != current_commit:
                        continue
                    if record.created_at and record.created_at < cutoff_naive:
                        continue
                    return (record, embedding)

        return (None, embedding if qa_ids is not None else None)

    async def record_interaction(
        self,
        qa_id: str,
        wiki_id: str,
        question: str,
        answer: str,
        sources_json: str,
        tool_steps: int,
        mode: str,
        user_id: str | None,
        is_cache_hit: bool,
        source_qa_id: str | None,
        embedding: np.ndarray | None,
        has_context: bool = False,
        source_commit_hash: str | None = None,
    ) -> None:
        """Persist a QA interaction and optionally update the FAISS index."""
        question_hash = _compute_question_hash(question)
        record = QARecord(
            id=qa_id,
            wiki_id=wiki_id,
            question=question,
            question_hash=question_hash,
            answer=answer,
            sources_json=sources_json,
            tool_steps=tool_steps,
            mode=mode,
            user_id=user_id,
            is_cache_hit=1 if is_cache_hit else 0,
            source_qa_id=source_qa_id,
            has_context=1 if has_context else 0,
            source_commit_hash=source_commit_hash,
        )
        try:
            async with self._session_factory() as session:
                session.add(record)
                await session.commit()
        except Exception:
            logger.error("Failed to persist QA record %s", qa_id, exc_info=True)
            return  # No point indexing if the DB write failed

        # Add to FAISS index if: not a cache hit, not contextual, and embedding is available
        if not is_cache_hit and not has_context and embedding is not None and self._cache is not None:
            try:
                await self._cache.add(wiki_id, qa_id, embedding)
            except Exception:
                logger.error("Failed to index QA record %s in FAISS (DB write succeeded)", qa_id, exc_info=True)

    async def list_qa(
        self,
        wiki_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[QARecord], int]:
        """List QA records with pagination and optional status filter."""
        async with self._session_factory() as session:
            base = select(QARecord).where(QARecord.wiki_id == wiki_id)
            if status:
                base = base.where(QARecord.status == status)

            # Get total count
            count_stmt = select(func.count()).select_from(base.subquery())
            total = (await session.execute(count_stmt)).scalar() or 0

            # Get paginated records
            stmt = base.order_by(QARecord.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(stmt)
            records = list(result.scalars().all())

            return records, total

    async def get_stats(self, wiki_id: str) -> dict:
        """Get QA statistics for a wiki in a single query."""
        async with self._session_factory() as session:
            stmt = select(
                func.count().label("total"),
                func.sum(case((QARecord.is_cache_hit == 1, 1), else_=0)).label("cache_hits"),
                func.sum(case((QARecord.status == "validated", 1), else_=0)).label("validated"),
                func.sum(case((QARecord.status == "rejected", 1), else_=0)).label("rejected"),
            ).where(QARecord.wiki_id == wiki_id)

            row = (await session.execute(stmt)).one()
            total = row.total or 0
            cache_hit_count = row.cache_hits or 0

            return {
                "total_count": total,
                "cache_hit_count": cache_hit_count,
                "validated_count": row.validated or 0,
                "rejected_count": row.rejected or 0,
                "hit_rate": cache_hit_count / total if total > 0 else 0.0,
            }

    async def invalidate_wiki(self, wiki_id: str, delete: bool = False) -> None:
        """Invalidate QA data for a wiki (on delete or refresh)."""
        async with self._session_factory() as session:
            if delete:
                await session.execute(sa_delete(QARecord).where(QARecord.wiki_id == wiki_id))
            else:
                await session.execute(
                    sa_update(QARecord)
                    .where(
                        QARecord.wiki_id == wiki_id,
                        QARecord.status == "pending",
                    )
                    .values(status="stale")
                )
            await session.commit()

        if self._cache:
            self._cache.delete_index(wiki_id)
