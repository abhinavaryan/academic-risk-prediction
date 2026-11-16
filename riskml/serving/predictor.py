"""Validate a student's features against the schema and explain the model's risk score.

Explanations use *baseline replacement*: for each feature that differs from the typical
student (training median / most common value), we re-score the student with that one feature
set to the baseline. The change in risk is that feature's impact. It is model-agnostic,
exact for the deployed model, and easy to explain in plain language.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from riskml.config import BAND_HIGH, BAND_MODERATE
from riskml.serving.registry import LoadedModel

MAX_FACTORS = 6


class FeatureValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


@dataclass(frozen=True)
class Factor:
    name: str
    label: str
    value: Any
    baseline: Any
    impact: float
    direction: str


@dataclass(frozen=True)
class Prediction:
    cutoff: int
    probability: float
    band: str
    at_risk: bool
    threshold: float
    factors: tuple[Factor, ...]
    filled_with_baseline: tuple[str, ...]


def band_of(p: float) -> str:
    if p >= BAND_HIGH:
        return "high"
    if p >= BAND_MODERATE:
        return "moderate"
    return "low"


def validate_features(schema: dict, features: Mapping[str, Any]) -> dict[str, Any]:
    spec = {f["name"]: f for f in schema["features"]}
    errors: dict[str, str] = {}
    clean: dict[str, Any] = {}
    for name, value in features.items():
        f = spec.get(name)
        if f is None:
            errors[name] = "unknown feature"
            continue
        if value is None:
            if f.get("nullable"):
                clean[name] = None
            else:
                errors[name] = "is required and cannot be null"
            continue
        if f["kind"] == "categorical":
            if not isinstance(value, str) or value not in f["levels"]:
                errors[name] = "must be one of: " + ", ".join(f["levels"])
            else:
                clean[name] = value
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors[name] = "must be a number"
        elif not math.isfinite(value):
            errors[name] = "must be a finite number"
        elif value < f["min"] or value > f["max"]:
            errors[name] = f"must be between {f['min']:g} and {f['max']:g}"
        else:
            clean[name] = float(value)
    if errors:
        raise FeatureValidationError(errors)
    return clean


def _frame(loaded: LoadedModel, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=loaded.feature_names)
    for f in loaded.schema["features"]:
        if f["kind"] == "numeric":
            df[f["name"]] = pd.to_numeric(df[f["name"]], errors="coerce").astype("float64")
        else:
            df[f["name"]] = df[f["name"]].astype("object").where(df[f["name"]].notna(), np.nan)
    return df


def predict(loaded: LoadedModel, features: Mapping[str, Any]) -> Prediction:
    clean = validate_features(loaded.schema, features)
    row = {name: clean.get(name, loaded.baseline[name]) for name in loaded.feature_names}
    filled = tuple(n for n in loaded.feature_names if n not in clean)
    differing = [n for n in loaded.feature_names
                 if n in clean and clean[n] != loaded.baseline[n]]
    rows = [row] + [{**row, n: loaded.baseline[n]} for n in differing]
    probs = loaded.model.predict_proba(_frame(loaded, rows))[:, 1]
    p = float(probs[0])
    factors = []
    for n, p_base in zip(differing, probs[1:], strict=True):
        impact = round(p - float(p_base), 4)
        if impact != 0:
            factors.append(Factor(
                name=n, label=loaded.labels[n], value=row[n], baseline=loaded.baseline[n],
                impact=impact, direction="increases" if impact > 0 else "decreases"))
    factors.sort(key=lambda f: abs(f.impact), reverse=True)
    return Prediction(
        cutoff=loaded.cutoff, probability=round(p, 4), band=band_of(p),
        at_risk=p >= loaded.threshold, threshold=loaded.threshold,
        factors=tuple(factors[:MAX_FACTORS]), filled_with_baseline=filled,
    )
