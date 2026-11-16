"""Project-wide constants. Every module reads limits and paths from here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("AGRIM_ARTIFACTS_DIR", ROOT / "artifacts"))
GENERATED_DIR = ROOT / "web" / "src" / "generated"
REPORTS_DIR = ROOT / "reports"
SAMPLE_DATA_DIR = ROOT / "sample_data"
DEFAULT_DATA_DIR = ROOT / "data" / "oulad"

SEED = 42

# Prediction days (days since module start). Day 60 is the pre-registered primary cutoff.
CUTOFFS: tuple[int, ...] = (0, 30, 60, 90, 120, 150, 180)
PRIMARY_CUTOFF = 60

# Temporal split: train on earlier presentations, test on the most recent one.
TEST_PRESENTATION = "2014J"
AT_RISK_RESULTS = frozenset({"Fail", "Withdrawn"})

# Official OULAD mirror (UCI ML Repository, CC BY 4.0) and its SHA-256.
OULAD_URL = (
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
)
OULAD_SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"

# Risk bands used everywhere in the product.
BAND_MODERATE = 0.33
BAND_HIGH = 0.66

# Bring-your-own-data limits (Vercel caps request bodies at 4.5 MB).
MAX_UPLOAD_BYTES = 4_000_000
MAX_ROWS = 50_000
MAX_COLS = 200
MAX_XLSX_UNCOMPRESSED = 60_000_000
MIN_LABELED_ROWS = 30
MIN_CLASS_COUNT = 10
CUSTOM_TIME_BUDGET_S = float(os.environ.get("AGRIM_TIME_BUDGET_S", "40"))
MAX_CONCURRENT_TRAININGS = int(os.environ.get("AGRIM_MAX_TRAININGS", "2"))
CUSTOM_RATE_PER_MIN = int(os.environ.get("AGRIM_RATE_PER_MIN", "20"))
MAX_REASON_ROWS = 2_000
