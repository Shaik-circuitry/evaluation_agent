"""Configuration: Portkey, weights, thresholds."""

import os

from dotenv import load_dotenv

load_dotenv()

# Portkey (OpenAI-compatible gateway)
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "Yi21oetWtITsPYeqomKdh+eq+HMP")
PORTKEY_CONFIG = os.getenv("PORTKEY_CONFIG", "pc-intern-6ab2e3")
PORTKEY_BASE_URL = os.getenv("PORTKEY_BASE_URL", "https://api.portkey.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))

# Backward-compatible name used in reports / eval_model
MODEL = LLM_MODEL

DIMENSIONS = {
    "semantic_similarity": {
        "weight": 0.25,
        "description": "How closely does the AI response meaning align with the human response?",
    },
    "factual_accuracy": {
        "weight": 0.30,
        "description": "Are all facts, procedures, and claims correct? Any hallucination = score 1.",
    },
    "completeness": {
        "weight": 0.20,
        "description": "Does the AI response cover all aspects present in the human response?",
    },
    "tone_professionalism": {
        "weight": 0.10,
        "description": "Is tone professional, empathetic where needed, and readable?",
    },
}

# Composite: weighted avg of dim scores (each 1–5), normalized to 0–100
# Weights sum to 0.85; remaining 15% reserved for future actionability dimension.

GO_LIVE_THRESHOLD = 80
RAG_DIAGNOSIS_THRESHOLD = 70
HALLUCINATION_SCORE = 1
