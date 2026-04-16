"""Pretty-print single results and aggregate summaries."""

from collections import Counter

from config import DIMENSIONS, GO_LIVE_THRESHOLD, RAG_DIAGNOSIS_THRESHOLD

from evaluator.models import EvaluationResult
from evaluator.store import EvaluationStore


def print_result(result: EvaluationResult) -> None:
    print("=" * 72)
    print(f"Evaluation record: {result.record_id}")
    print(f"Model: {result.eval_model}  |  Created: {result.created_at.isoformat()}")
    print("-" * 72)
    q = result.input.query
    snippet = q if len(q) <= 200 else q[:197] + "..."
    print("Query:")
    print(snippet)
    print("-" * 72)
    print("Dimension scores:")
    for name, ds in result.dimension_scores.items():
        w = DIMENSIONS.get(name, {}).get("weight", "?")
        print(f"  [{name}] (weight {w}): {ds.score}/5")
        print(f"    {ds.rationale}")
    print("-" * 72)
    print(f"Composite score: {result.composite_score:.2f} / 100")
    print(f"Flags: {result.flags if result.flags else '(none)'}")
    print("-" * 72)
    print("Overall rationale:")
    print(result.overall_rationale)
    if result.rag_diagnosis:
        print("-" * 72)
        print("RAG diagnosis (below threshold path):")
        print(result.rag_diagnosis)
    print("=" * 72)


def print_summary(store: EvaluationStore) -> None:
    all_r = store.get_all()
    n = len(all_r)
    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)
    print(f"Total evaluations: {n}")
    if n == 0:
        print("(No records.)")
        print("=" * 72)
        return

    avg_comp = sum(r.composite_score for r in all_r) / n
    print(f"Average composite score: {avg_comp:.2f}")

    dim_names = list(DIMENSIONS.keys())
    for d in dim_names:
        scores = [r.dimension_scores[d].score for r in all_r if d in r.dimension_scores]
        if scores:
            print(f"  Average {d}: {sum(scores) / len(scores):.2f} / 5")

    below_go = [r for r in all_r if r.composite_score < GO_LIVE_THRESHOLD]
    verdict = "GO" if avg_comp >= GO_LIVE_THRESHOLD and not below_go else "NO-GO"
    if below_go and avg_comp >= GO_LIVE_THRESHOLD:
        verdict = "NO-GO (some records below threshold)"
    print("-" * 72)
    print(f"Go/no-go threshold: {GO_LIVE_THRESHOLD} (based on average and per-record checks)")
    print(f"Verdict (heuristic): {verdict}")

    flag_counter: Counter[str] = Counter()
    for r in all_r:
        for f in r.flags:
            flag_counter[f] += 1
    print("-" * 72)
    print("Flag counts (records may have multiple flags):")
    for flag, cnt in sorted(flag_counter.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {cnt}")
    if not flag_counter:
        print("  (none)")

    low_ids = [
        r.record_id
        for r in all_r
        if r.composite_score < RAG_DIAGNOSIS_THRESHOLD
    ]
    print("-" * 72)
    print(
        f"Records with composite < {RAG_DIAGNOSIS_THRESHOLD} (needs KB / retrieval attention):"
    )
    if low_ids:
        for rid in low_ids:
            print(f"  - {rid}")
    else:
        print("  (none)")
    print("=" * 72)
