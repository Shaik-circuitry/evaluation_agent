# RAG Advisor Evaluation System

Python library for benchmarking a RAG-based AI Advisor against human responses using **LLM-as-judge**, **composite scoring**, optional **RAG failure diagnostics**, and **in-memory** storage. No HTTP server or CLI required—import and call functions directly.

## Setup

1. Create a virtual environment (recommended).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your Anthropic API key:

   - **Recommended:** create a `.env` file in `rag_evaluator/` with:

     ```
     ANTHROPIC_API_KEY=sk-ant-...
     ```

   - Or set the environment variable `ANTHROPIC_API_KEY`.

   - `config.py` reads the key via `python-dotenv` and `os.environ`.

4. Model name is set in `config.py` as `MODEL` (default: `claude-sonnet-4-20250514`).

## Run the demo

From the `rag_evaluator` directory:

```bash
python demo_cli.py
```

Edit the `# --- INPUT VARIABLES ---` section at the top of `demo_cli.py` to try different queries and responses.

## Use as a library

```python
from evaluator import run_evaluation
from evaluator.models import EvaluationInput, RetrievedChunk
from evaluator.report import print_result, print_summary
from evaluator import get_store

inp = EvaluationInput(
    query="Your question",
    human_response="...",
    rag_response="...",
    retrieved_chunks=[RetrievedChunk(text="...", relevance_score=0.7)],
)
result = run_evaluation(inp)
print_result(result)
print_summary(get_store())
```

`run_evaluation` uses a **module-level** `EvaluationStore`, so repeated calls in the same Python process accumulate records for `print_summary`.

## Scoring dimensions and weights

| Dimension | Weight | Meaning |
|-----------|--------|---------|
| `semantic_similarity` | 0.25 | How closely the AI answer aligns in meaning with the human reference. |
| `factual_accuracy` | 0.30 | Correctness of facts/procedures; unsupported claims → low score (1 = hallucination risk). |
| `completeness` | 0.20 | Whether the AI covers the aspects present in the human response. |
| `tone_professionalism` | 0.10 | Professional, empathetic, readable tone. |

Weights sum to **0.85**; the remaining **15%** is reserved for a future dimension (e.g. actionability).

Each dimension is scored **1–5** by the judge. The **composite score** is:

\[
\text{composite} = \frac{\sum_i w_i \cdot s_i}{5 \cdot \sum_i w_i} \times 100
\]

where \(s_i\) is the score for dimension \(i\) and \(w_i\) is its weight. Result is on **0–100** (two decimal places).

## Go / no-go

- **`GO_LIVE_THRESHOLD`** (default **80**) in `config.py`: average or per-record policy is up to you; the **report** prints a heuristic verdict comparing aggregates to this threshold.

## RAG diagnosis

If the composite score is **below `RAG_DIAGNOSIS_THRESHOLD`** (default **70**), a **second** Anthropic call runs `analyze_rag_failure` in `evaluator/rag_diagnostics.py`, which inspects the query, RAG answer, human reference, retrieved chunks, and relevance scores, and returns a short paragraph on retrieval vs. coverage vs. utilization issues.

## Flags

The judge may emit flags such as `hallucination`, `incomplete`, `tone_mismatch`, `superior_to_rag`. Additional rules enforce flags when factual score is 1, or completeness/tone are ≤2.

## Project layout

- `config.py` — API key (via env), model id, dimensions, thresholds  
- `evaluator/models.py` — Pydantic models  
- `evaluator/judge.py` — LLM-as-judge + position randomization  
- `evaluator/scorer.py` — composite calculation  
- `evaluator/rag_diagnostics.py` — low-score RAG analysis  
- `evaluator/store.py` — in-memory store  
- `evaluator/report.py` — `print_result`, `print_summary`  
- `evaluator/__init__.py` — `run_evaluation`, `get_store`  
- `demo_cli.py` — CLI demo  
- `web_app.py` / `app.py` — Flask UI + API (`app.py` is the Vercel WSGI entry)

## Deploy on Vercel

1. **Root directory:** If the Git repo root is above `rag_evaluator`, set **Project → Settings → Root Directory** to `rag_evaluator`.
2. **Environment variables** (match `config.py`): e.g. `PORTKEY_API_KEY`, `PORTKEY_CONFIG`, `PORTKEY_BASE_URL`, `LLM_MODEL`, etc.
3. Deploy: connect the repo in Vercel or run `vercel` from `rag_evaluator` (CLI ≥ 48.2.10).
4. **Timeouts:** `vercel.json` sets `maxDuration` to **120s** for the Flask function (judge calls). On Hobby, serverless limits may be lower—upgrade or shorten judge timeout if deploy fails.
5. **Health check:** `GET /health` returns `{"status":"ok"}`.

`main.py` was renamed to `demo_cli.py` so Vercel does not treat it as a second Python entrypoint.
