"""Vercel serverless entry point: exposes the FastAPI app as a Python function."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskml.server.main import app

__all__ = ["app"]
