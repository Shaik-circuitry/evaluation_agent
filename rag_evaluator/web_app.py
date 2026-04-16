"""Serve the RAG evaluation web UI and API."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from evaluator import run_evaluation
from evaluator.models import EvaluationInput, RetrievedChunk

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    """Liveness/readiness probe for Kubernetes (EKS)."""
    return jsonify({"status": "ok"}), 200


@app.post("/api/evaluate")
def api_evaluate():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object"}), 400

        chunks_raw = data.get("retrieved_chunks") or []
        if not isinstance(chunks_raw, list):
            return jsonify({"error": "retrieved_chunks must be an array"}), 400

        chunks = []
        for i, c in enumerate(chunks_raw):
            if not isinstance(c, dict):
                return jsonify({"error": f"Chunk {i} must be an object"}), 400
            chunks.append(RetrievedChunk(**c))

        inp = EvaluationInput(
            query=str(data.get("query", "")),
            human_response=str(data.get("human_response", "")),
            rag_response=str(data.get("rag_response", "")),
            retrieved_chunks=chunks,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )
        result = run_evaluation(inp)
        payload = result.model_dump(mode="json")
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/api/demo")
def api_demo():
    print("[web_app] /api/demo: Demo handler invoked from the UI.", flush=True)
    return jsonify({"message": "Server logged a demo line; check the terminal running web_app.py."})


def main():
    import os

    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5050"))
    print(f"Open http://{host}:{port} — POST /api/evaluate and POST /api/demo", flush=True)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
