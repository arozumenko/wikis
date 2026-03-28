"""Q&A Knowledge Flywheel API models and internal dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.models.api import AskResponse

if TYPE_CHECKING:
    import numpy as np

# Valid status values for QARecord.status
QAStatus = Literal["pending", "validated", "rejected", "enriched", "stale"]

# Valid mode values for QARecord.mode
QAMode = Literal["fast", "deep"]


@dataclass
class QARecordingPayload:
    """Data needed to persist a QA interaction. Created by AskService, consumed by callers."""
    qa_id: str
    wiki_id: str
    question: str
    answer: str
    sources_json: str
    tool_steps: int
    mode: QAMode
    user_id: str | None
    is_cache_hit: bool
    source_qa_id: str | None
    embedding: np.ndarray | None
    has_context: bool = False
    source_commit_hash: str | None = None


@dataclass
class AskResult:
    """Returned by AskService.ask_sync. Callers handle recording per their context."""
    response: AskResponse
    recording: QARecordingPayload | None


class QARecordResponse(BaseModel):
    """Single QA record in API responses."""
    id: str
    wiki_id: str
    question: str
    answer: str
    sources_json: str | None = None
    tool_steps: int = 0
    mode: QAMode = "fast"
    status: QAStatus = "pending"
    is_cache_hit: bool = False
    has_context: bool = False
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class QAListResponse(BaseModel):
    """Paginated QA list response."""
    items: list[QARecordResponse]
    total: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)


class QAStatsResponse(BaseModel):
    """QA statistics response."""
    total_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    validated_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    hit_rate: float = Field(ge=0.0, le=1.0)
