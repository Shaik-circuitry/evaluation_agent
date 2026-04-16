
"""
Vercel WSGI entrypoint.

Vercel's Python runtime loads a top-level `app` from app.py / index.py / server.py.
This module re-exports the Flask application defined in web_app.py (same process,
same routes: /, /api/evaluate, /health, /api/demo).
"""
from web_app import app

__all__ = ["app"]
