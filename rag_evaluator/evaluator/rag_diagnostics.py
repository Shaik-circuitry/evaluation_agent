"""Second-pass RAG failure analysis when composite score is below threshold."""

from __future__ import annotations

from config import PORTKEY_API_KEY
from llm_client import chat_completion
from evaluator.models import EvaluationInput, EvaluationResult


def analyze_rag_failure(input: EvaluationInput, eval_result: EvaluationResult) -> str:
    """
    Dedicated LLM call (via Portkey) to diagnose retrieval / utilization issues.
    """
    if not PORTKEY_API_KEY:
        raise RuntimeError("PORTKEY_API_KEY is not set.")

    chunks_lines: list[str] = []
    for i, ch in enumerate(input.retrieved_chunks, start=1):
        rs = ch.relevance_score
        rs_s = f"{rs:.2f}" if rs is not None else "n/a"
        chunks_lines.append(f"Chunk {i} (relevance_score={rs_s}):\n{ch.text}")
    chunks_block = "\n\n".join(chunks_lines) if chunks_lines else "(No retrieved chunks.)"

    dim_summary = []
    for name, ds in eval_result.dimension_scores.items():
        dim_summary.append(f"- {name}: {ds.score}/5 — {ds.rationale}")
    dim_block = "\n".join(dim_summary)

    user = f"""You are a retrieval and RAG pipeline analyst. Diagnose why the advisor response may be failing.

Customer query:
{input.query}

RAG (advisor) response:
{input.rag_response}

Reference (human) response (for gap comparison only):
{input.human_response}

Retrieved chunks:
{chunks_block}

Prior evaluation summary (composite {eval_result.composite_score:.2f}/100):
{dim_block}

Analyze concisely (3–5 sentences, one short paragraph):
1. Query–chunk relevance gap: Are retrieved chunks aligned with query intent?
2. Coverage gap: Does the union of chunks contain what is needed to answer correctly? What is missing?
3. Utilization: If chunks had the right information, did the RAG answer fail to use them?
4. Relevance scores: If provided, flag scores < 0.5 as suspicious for embedding/retrieval quality.

Name the primary failure mode (e.g. wrong retrieval, missing KB content, generation ignored context) and one concrete fix (KB, chunking, reranker, or prompt). Output plain text only, no JSON."""

    return chat_completion(
        system="You output only the diagnostic paragraph, no preamble.",
        user=user,
        max_tokens=800,
    )
