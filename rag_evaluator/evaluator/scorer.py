"""Composite score from dimension scores and config weights."""

from config import DIMENSIONS

from evaluator.models import DimensionScore


def calculate_composite(dimension_scores: dict[str, DimensionScore]) -> float:
    """
    Weighted average of 1–5 dimension scores, normalized to 0–100.

    Formula: sum(weight * score) / (5 * sum_of_weights) * 100
    """
    sum_weights = sum(meta["weight"] for meta in DIMENSIONS.values())
    weighted = 0.0
    for dim_name, meta in DIMENSIONS.items():
        if dim_name not in dimension_scores:
            raise KeyError(f"Missing dimension for composite: {dim_name}")
        weighted += meta["weight"] * dimension_scores[dim_name].score
    denom = 5.0 * sum_weights
    if denom <= 0:
        raise ValueError("Sum of weights must be positive.")
    return round((weighted / denom) * 100.0, 2)
