"""SQLAlchemy ORM models for persistent wiki metadata storage."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class WikiRecord(Base):
    """Persistent record of a generated wiki and its ownership metadata."""

    __tablename__ = "wiki"

    id = Column(String, primary_key=True)  # wiki_id e.g. "owner--repo--main"
    owner_id = Column(String, nullable=False, index=True)
    repo_url = Column(String, nullable=False)
    branch = Column(String, nullable=False)
    title = Column(String, nullable=False)
    page_count = Column(Integer, default=0)
    visibility = Column(String, default="personal")  # "personal" or "shared"
    commit_hash = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    indexed_at = Column(DateTime, nullable=True)
    status = Column(String, default="generating")  # generating | complete | failed | partial | cancelled
    requires_token = Column(Integer, default=0)  # 1 if repo required an access token (Boolean as int for SQLite)
    error = Column(String, nullable=True)  # error message for failed generations
    description = Column(String, nullable=True)  # user-editable wiki description

    __table_args__ = (Index("ix_wiki_owner_visibility", "owner_id", "visibility"),)


class WikiPageRecord(Base):
    """Indexed wiki page for full-text search.

    Backend-aware FTS:
    - PostgreSQL: a STORED ``search_vector`` (tsvector) column is appended via
      :func:`app.db._ensure_wiki_page_fts` and queried with ``ts_rank``.
      Weights: title=A, description=B, content=C.
    - SQLite: a ``wiki_page_fts`` FTS5 virtual table is created with the
      same three columns plus sync triggers; queried via ``bm25()`` with the
      equivalent column weights.
    """

    __tablename__ = "wiki_page"

    id = Column(String, primary_key=True)          # "{wiki_id}/{page_title}" deterministic
    wiki_id = Column(String, nullable=False)
    page_title = Column(String, nullable=False)
    description = Column(String, nullable=True)     # From YAML frontmatter
    content = Column(Text, nullable=False)          # Full markdown body

    __table_args__ = (Index("ix_wiki_page_wiki", "wiki_id"),)


class ProjectRecord(Base):
    """Persistent record of a project grouping wikis together."""

    __tablename__ = "project"

    id = Column(String, primary_key=True)  # UUID4
    owner_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    visibility = Column(String, default="personal")  # "personal" | "shared"
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_project_owner_visibility", "owner_id", "visibility"),)


class ProjectWikiRecord(Base):
    """Association between a project and a wiki."""

    __tablename__ = "project_wiki"

    project_id = Column(String, nullable=False, primary_key=True)
    wiki_id = Column(String, nullable=False, primary_key=True)
    added_at = Column(DateTime, server_default=func.now())
    added_by = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_project_wiki_project", "project_id"),
        Index("ix_project_wiki_wiki", "wiki_id"),
    )


class QARecord(Base):
    """Persistent record of a Q&A interaction for the knowledge flywheel."""

    __tablename__ = "qa_record"

    id = Column(String, primary_key=True)              # Pre-generated UUID
    wiki_id = Column(String, nullable=False)           # No FK (matches WikiRecord pattern)
    question = Column(String, nullable=False)
    question_hash = Column(String, nullable=False)     # SHA256 hex digest
    answer = Column(String, nullable=False)
    sources_json = Column(String, nullable=True)       # JSON list[SourceReference]
    tool_steps = Column(Integer, default=0)
    mode = Column(String, default="fast")              # "fast" | "deep"
    status = Column(String, default="pending")         # pending|validated|rejected|enriched|stale
    validation_type = Column(String, nullable=True)    # null|user_upvote|user_downvote
    is_cache_hit = Column(Integer, default=0)          # SQLite-compatible boolean (0/1)
    source_qa_id = Column(String, nullable=True)       # UUID of original record for cache hits
    has_context = Column(Integer, default=0)           # 1 if chat_history was non-empty
    source_commit_hash = Column(String, nullable=True) # Wiki commit_hash at recording time
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    validated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_qa_wiki_hash", "wiki_id", "question_hash"),
        Index("ix_qa_wiki_status", "wiki_id", "status"),
        Index("ix_qa_wiki_cache", "wiki_id", "is_cache_hit"),
    )
