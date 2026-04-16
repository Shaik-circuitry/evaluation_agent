# RAG evaluation service — Flask UI + /api/evaluate
#
# Build (from repo root: evaluation_agent):
#   docker build -f rag_evaluator/Dockerfile -t rag-evaluator:latest rag_evaluator
#
# Run locally:
#   docker run --rm -p 8080:8080 -e PORTKEY_API_KEY=... -e PORTKEY_CONFIG=... rag-evaluator:latest
#
# Container filesystem layout:
#   /app/web_app.py          — Flask app (gunicorn entry: web_app:app)
#   /app/evaluator/          — evaluation pipeline
#   /app/web/                — static UI (index.html)
#   /app/config.py, etc.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    WEB_CONCURRENCY=2

# Non-root user (common EKS / pod security policy)
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --shell /usr/sbin/nologin --create-home app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=10001:app . .

USER 10001:10001

EXPOSE 8080

# PORT / WEB_CONCURRENCY can be overridden at deploy time (K8s sets PORT in some setups; we default 8080).
# --timeout: allow long LLM judge calls; tune with your gateway limits.
CMD ["sh", "-c", "exec gunicorn \
  --bind 0.0.0.0:${PORT:-8080} \
  --workers ${WEB_CONCURRENCY:-2} \
  --threads 1 \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile - \
  web_app:app"]
