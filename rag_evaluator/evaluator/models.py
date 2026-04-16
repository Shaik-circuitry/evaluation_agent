"""Pydantic data models for RAG evaluation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    relevance_score: float | None = None


class EvaluationInput(BaseModel):
    query: str
    human_response: str
    rag_response: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DimensionScore(BaseModel):
    score: int
    rationale: str


class EvaluationResult(BaseModel):
    record_id: str
    input: EvaluationInput
    dimension_scores: dict[str, DimensionScore]
    composite_score: float
    flags: list[str]
    overall_rationale: str
    rag_diagnosis: str | None
    eval_model: str
    created_at: datetime
