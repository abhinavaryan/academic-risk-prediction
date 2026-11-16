"""Load the seven OULAD tables and apply the dataset-specific cleaning rules.

Cleaning rules (documented in docs/METHODOLOGY.md):
* '?' marks a missing value in the raw CSVs -> NaN.
* imd_band '10-20' is missing its '%' sign in the source -> '10-20%'.
* studentVle is joined with vle.activity_type so features can count clicks per activity type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
COURSE_KEYS = ["code_module", "code_presentation"]

_FILES = {
    "courses": "courses.csv",
    "info": "studentInfo.csv",
    "registration": "studentRegistration.csv",
    "assessments": "assessments.csv",
    "student_assessment": "studentAssessment.csv",
    "vle": "vle.csv",
    "student_vle": "studentVle.csv",
}

_NUMERIC = {
    "courses": ["module_presentation_length"],
    "info": ["id_student", "num_of_prev_attempts", "studied_credits"],
    "registration": ["id_student", "date_registration", "date_unregistration"],
    "assessments": ["id_assessment", "date", "weight"],
    "student_assessment": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    "vle": ["id_site"],
    "student_vle": ["id_student", "id_site", "date", "sum_click"],
}


@dataclass(frozen=True)
class OuladTables:
    courses: pd.DataFrame
    info: pd.DataFrame
    registration: pd.DataFrame
    assessments: pd.DataFrame
    student_assessment: pd.DataFrame
    vle: pd.DataFrame
    student_vle: pd.DataFrame


def _clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    for col in _NUMERIC[name]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_tables(raw: dict[str, pd.DataFrame]) -> OuladTables:
    frames = {name: _clean(name, raw[name].copy()) for name in _FILES}
    info = frames["info"]
    info["imd_band"] = info["imd_band"].replace({"10-20": "10-20%"})
    vle = frames["vle"][["id_site", "activity_type"]].drop_duplicates("id_site")
    sv = frames["student_vle"].merge(vle, on="id_site", how="left")
    sv["activity_type"] = sv["activity_type"].fillna("unknown")
    sv["date"] = sv["date"].astype("int32")
    sv["sum_click"] = sv["sum_click"].astype("int32")
    sv["id_student"] = sv["id_student"].astype("int64")
    frames["student_vle"] = sv
    return OuladTables(**frames)


def load_tables(data_dir: Path, cache_dir: Path | None = None) -> OuladTables:
    """Read the raw CSVs (studentVle is ~10.6M rows; cached as parquet when cache_dir is set)."""
    raw: dict[str, pd.DataFrame] = {}
    for name, filename in _FILES.items():
        if name == "student_vle" and cache_dir is not None:
            cached = cache_dir / "studentVle.parquet"
            if cached.exists():
                raw[name] = pd.read_parquet(cached)
                continue
            df = pd.read_csv(
                data_dir / filename,
                dtype={"code_module": "category", "code_presentation": "category"},
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cached, index=False)
            raw[name] = df
            continue
        raw[name] = pd.read_csv(data_dir / filename, na_values="?", keep_default_na=True)
    for name in ("student_vle",):
        for col in ("code_module", "code_presentation"):
            raw[name][col] = raw[name][col].astype(str)
    return clean_tables(raw)
