"""RAG advisor evaluation: judge, score, optional RAG diagnostics, in-memory store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from config import MODEL, RAG_DIAGNOSIS_THRESHOLD

from evaluator.judge import run_judge
from evaluator.models import DimensionScore, EvaluationInput, EvaluationResult
from evaluator.rag_diagnostics import analyze_rag_failure
from evaluator.scorer import calculate_composite
from evaluator.store import EvaluationStore

_store = EvaluationStore()


def get_store() -> EvaluationStore:
    """Access the module-level evaluation store (same session lifetime)."""
    return _store


def run_evaluation(input: EvaluationInput) -> EvaluationResult:
    """
    Run LLM judge → composite score → optional RAG diagnostics → persist → return.
    """
    raw = run_judge(input)

    dimension_scores: dict[str, DimensionScore] = {}
    for name, payload in raw["dimension_scores"].items():
        dimension_scores[name] = DimensionScore(
            score=int(payload["score"]),
            rationale=str(payload["rationale"]),
        )

    composite = calculate_composite(dimension_scores)

    record_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    result = EvaluationResult(
        record_id=record_id,
        input=input,
        dimension_scores=dimension_scores,
        composite_score=composite,
        flags=list(raw["flags"]),
        overall_rationale=str(raw["overall_rationale"]),
        rag_diagnosis=None,
        eval_model=str(raw.get("eval_model", MODEL)),
        created_at=created_at,
    )

    if composite < RAG_DIAGNOSIS_THRESHOLD:
        diagnosis = analyze_rag_failure(input, result)
        result = result.model_copy(update={"rag_diagnosis": diagnosis})

    _store.add(result)
    return result


__all__ = [
    "run_evaluation",
    "get_store",
    "EvaluationInput",
    "EvaluationResult",
]
