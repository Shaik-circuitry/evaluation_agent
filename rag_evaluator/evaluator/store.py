"""In-memory storage for evaluation records."""

from evaluator.models import EvaluationResult


class EvaluationStore:
    def __init__(self) -> None:
        self._by_id: dict[str, EvaluationResult] = {}

    def add(self, result: EvaluationResult) -> None:
        self._by_id[result.record_id] = result

    def get(self, record_id: str) -> EvaluationResult | None:
        return self._by_id.get(record_id)

    def get_all(self) -> list[EvaluationResult]:
        return list(self._by_id.values())

    def get_flagged(self) -> list[EvaluationResult]:
        return [r for r in self._by_id.values() if r.flags]

    def clear(self) -> None:
        self._by_id.clear()
