"""LLM-as-Judge: Portkey (OpenAI-compatible), position randomization, structured JSON output."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from config import DIMENSIONS, HALLUCINATION_SCORE, MODEL, PORTKEY_API_KEY
from llm_client import chat_completion
from evaluator.models import EvaluationInput


class JudgeError(Exception):
    """Raised when the judge returns invalid or unparseable output."""


def _build_rubric_block() -> str:
    lines = [
        "Scoring scale (each dimension 1–5):",
        "5 = Excellent: AI response is equivalent or superior to the reference on this dimension.",
        "4 = Good: Minor gaps but substantively meets the bar set by the reference.",
        "3 = Acceptable: Noticeable differences but core intent is preserved; improvement needed.",
        "2 = Below standard: Significant gaps, missing key information, or partial misalignment.",
        "1 = Unacceptable: Fundamentally wrong, incorrect, harmful, or hallucinates facts.",
        "",
        "Dimensions (score the ADVISOR / AI response only):",
    ]
    for name, meta in DIMENSIONS.items():
        lines.append(f"- {name} (weight {meta['weight']}): {meta['description']}")
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise JudgeError("Empty response from judge model.")
    # Strip optional markdown fence
    fence = re.match(r"^```(?:json)?\s*\n?", text)
    if fence:
        text = text[fence.end() :]
        text = re.sub(r"\n```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Invalid JSON from judge: {e}") from e


def _apply_flag_rules(
    dimension_scores: dict[str, Any],
    flags: list[str],
) -> list[str]:
    out = list(dict.fromkeys(flags))
    ds = dimension_scores
    fa = ds.get("factual_accuracy", {})
    fa_score = fa.get("score") if isinstance(fa, dict) else None
    if fa_score == HALLUCINATION_SCORE and "hallucination" not in out:
        out.append("hallucination")
    comp = ds.get("completeness", {})
    comp_score = comp.get("score") if isinstance(comp, dict) else None
    if comp_score is not None and comp_score <= 2 and "incomplete" not in out:
        out.append("incomplete")
    tone = ds.get("tone_professionalism", {})
    tone_score = tone.get("score") if isinstance(tone, dict) else None
    if tone_score is not None and tone_score <= 2 and "tone_mismatch" not in out:
        out.append("tone_mismatch")
    return out


def run_judge(input: EvaluationInput) -> dict[str, Any]:
    """
    Call the judge LLM. Returns a dict with:
    dimension_scores, flags, overall_rationale (rag_diagnosis not used — filled by diagnostics pass).
    """
    if not PORTKEY_API_KEY:
        raise JudgeError("PORTKEY_API_KEY is not set. Set it in the environment or config.")

    swap = random.random() < 0.5
    if swap:
        label_a, text_a = "Response A", input.human_response
        label_b, text_b = "Response B", input.rag_response
        mapping = "Response A is the REFERENCE (human SME) response. Response B is the AI ADVISOR response to score."
    else:
        label_a, text_a = "Response A", input.rag_response
        label_b, text_b = "Response B", input.human_response
        mapping = "Response A is the AI ADVISOR response to score. Response B is the REFERENCE (human SME) response."

    chunks_lines: list[str] = []
    for i, ch in enumerate(input.retrieved_chunks, start=1):
        rs = ch.relevance_score
        rs_s = f"{rs:.2f}" if rs is not None else "n/a"
        chunks_lines.append(f"Chunk {i} (relevance_score={rs_s}):\n{ch.text}")
    chunks_block = "\n\n".join(chunks_lines) if chunks_lines else "(No retrieved chunks provided.)"

    rubric = _build_rubric_block()

    system = (
        "You are an impartial evaluator for customer support quality. "
        "You must score only the AI ADVISOR response (identified below), not the reference. "
        "Respond with a single JSON object only — no markdown, no preamble, no trailing text."
    )

    user = f"""Customer query:
{input.query}

{label_a}:
{text_a}

{label_b}:
{text_b}

Mapping (for your scoring — do not reveal A/B labels in rationales; say "AI response" or "advisor"):
{mapping}

Retrieved context chunks (RAG retrieval for the advisor):
{chunks_block}

{rubric}

Instructions:
- Score the AI ADVISOR response on each dimension (1-5) with a short rationale per dimension.
- Use the reference response and retrieved chunks to judge factual accuracy and completeness of the AI response.
- If the AI states facts not supported by the reference or retrieved chunks, score factual_accuracy as 1 (hallucination risk).
- Flags (add to "flags" array as applicable): use "hallucination" if factual_accuracy is 1; "incomplete" if completeness score is 1 or 2; "tone_mismatch" if tone_professionalism is 1 or 2; "superior_to_rag" if the reference is clearly better overall than the AI (AI is materially worse).
- "overall_rationale": brief summary comparing AI to reference for stakeholders.
- Do NOT include rag_diagnosis in your output (always omit or null).

Required JSON shape (exact keys):
{{
  "dimension_scores": {{
    "semantic_similarity": {{"score": <1-5>, "rationale": "..."}},
    "factual_accuracy": {{"score": <1-5>, "rationale": "..."}},
    "completeness": {{"score": <1-5>, "rationale": "..."}},
    "tone_professionalism": {{"score": <1-5>, "rationale": "..."}}
  }},
  "flags": [],
  "overall_rationale": "..."
}}
"""

    raw = chat_completion(
        system=system,
        user=user,
        max_tokens=1500,
        response_format_json=True,
    )

    data = _parse_json_response(raw)

    required_keys = ("dimension_scores", "flags", "overall_rationale")
    for k in required_keys:
        if k not in data:
            raise JudgeError(f"Missing key in judge JSON: {k}")

    dim = data["dimension_scores"]
    for dname in DIMENSIONS:
        if dname not in dim:
            raise JudgeError(f"Missing dimension in judge output: {dname}")
        entry = dim[dname]
        if not isinstance(entry, dict) or "score" not in entry or "rationale" not in entry:
            raise JudgeError(f"Invalid structure for dimension: {dname}")
        sc = int(entry["score"])
        if sc < 1 or sc > 5:
            raise JudgeError(f"Score out of range for {dname}: {sc}")

    if not isinstance(data["flags"], list):
        raise JudgeError("'flags' must be a JSON array.")

    data["flags"] = _apply_flag_rules(dim, list(data["flags"]))
    data["eval_model"] = MODEL
    return data
