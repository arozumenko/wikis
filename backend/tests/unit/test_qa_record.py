"""Tests for QARecord data model."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.models.db_models import Base, QARecord


@pytest.fixture
def engine():
    """Create in-memory SQLite engine with tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_qa_record_instantiation():
    """QARecord can be created with all fields."""
    record = QARecord(
        id=str(uuid.uuid4()),
        wiki_id="wiki-123",
        question="What does this function do?",
        question_hash="abc123hash",
        answer="It processes data.",
        sources_json='[{"file_path": "main.py"}]',
        tool_steps=3,
        mode="fast",
        status="pending",
        is_cache_hit=0,
        has_context=0,
        source_commit_hash="deadbeef",
        user_id="user-1",
    )
    assert str(record.wiki_id) == "wiki-123"
    assert str(record.question_hash) == "abc123hash"
    assert int(record.is_cache_hit) == 0  # type: ignore[arg-type]
    assert int(record.has_context) == 0  # type: ignore[arg-type]
    assert str(record.status) == "pending"
    assert str(record.mode) == "fast"


def test_qa_record_table_creation(engine):
    """QARecord table is created via Base.metadata.create_all."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "qa_record" in tables


def test_qa_record_indexes(engine):
    """Composite indexes exist on qa_record table."""
    inspector = inspect(engine)
    indexes = inspector.get_indexes("qa_record")
    index_names = {idx["name"] for idx in indexes}
    assert "ix_qa_wiki_hash" in index_names
    assert "ix_qa_wiki_status" in index_names
    assert "ix_qa_wiki_cache" in index_names


def test_qa_record_insert_and_query(engine):
    """QARecord can be inserted and queried."""
    record_id = str(uuid.uuid4())
    with Session(engine) as session:
        record = QARecord(
            id=record_id,
            wiki_id="wiki-456",
            question="How does auth work?",
            question_hash="hash456",
            answer="It uses JWT tokens.",
            tool_steps=2,
            mode="deep",
        )
        session.add(record)
        session.commit()

        result = session.query(QARecord).filter_by(id=record_id).first()
        assert result is not None
        assert str(result.wiki_id) == "wiki-456"
        assert str(result.status) == "pending"  # default
        assert int(result.is_cache_hit) == 0  # type: ignore[arg-type]  # default
        assert int(result.has_context) == 0  # type: ignore[arg-type]  # default
