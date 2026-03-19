"""SQLAlchemy ORM models for persistent wiki metadata storage."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Index, Integer, String, func
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

    __table_args__ = (Index("ix_wiki_owner_visibility", "owner_id", "visibility"),)
