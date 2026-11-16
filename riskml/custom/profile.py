"""Column profiling for an uploaded dataset.

Each column gets a role:
* numeric      - used as a model feature (percent strings such as "85%" count as numbers)
* categorical  - used as a model feature (one-hot encoded)
* id           - unique per row (roll number, admission no.) -> used only to label rows
* pii          - names, e-mails, phone numbers -> never used as a feature
* text         - free text with too many distinct values -> not used
* constant / empty -> not used

It also suggests the target column and which of its values mean "at risk" / "result unknown".
"""

from __future__ import annotations

import re

import pandas as pd

_PCT = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*%\s*$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[\d\s()-]{10,15}$")
_ID_NAME = re.compile(
    r"(^id$|_id$|^id[_\s]|\bid\b|roll|enrol+ment\s*(no|number)|registration\s*(no|number)|"
    r"admission\s*(no|number)|student\s*(id|no|number)|\buid\b|serial|^s\.?\s?no\.?$|^sr\.?\s?no)",
    re.I)
_PII_NAME = re.compile(r"(name|e-?mail|phone|mobile|contact|address|aadha?r|guardian)", re.I)
_TARGET_NAME = re.compile(
    r"(target|result|outcome|status|label|class|final|pass|fail|dropout|grade|risk)", re.I)
_POSITIVE = re.compile(
    r"(fail|drop|withdr|detain|backlog|risk|reappear|atkt|defaulter|debar|discontinu|^f$)", re.I)
_UNKNOWN = re.compile(
    r"(enrol|ongoing|current|pending|in.?progress|awaited|running|^tbd$|^unknown$|^n/?a$)", re.I)

MAX_CATEGORIES = 50
MAX_TARGET_LEVELS = 10


def to_numeric(series: pd.Series) -> pd.Series:
    """Parse numbers, including percentages like '85%'."""
    s = series.astype("object")
    as_str = s.where(s.isna(), s.astype(str).str.strip())
    pct = as_str.str.match(_PCT, na=False)
    cleaned = as_str.where(~pct, as_str.str.replace("%", "", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def _role(name: str, s: pd.Series) -> str:
    non_null = s.dropna()
    n, n_unique = len(non_null), non_null.nunique()
    if n == 0:
        return "empty"
    if n_unique <= 1:
        return "constant"
    as_str = non_null.astype(str)
    if as_str.str.match(_EMAIL).mean() > 0.5:
        return "pii"
    numeric = to_numeric(non_null)
    numeric_share = numeric.notna().mean()
    all_unique = n_unique == n and n >= 20
    if _ID_NAME.search(name) and n_unique >= 0.9 * n:
        return "id"
    if numeric_share >= 0.95:
        if as_str.str.match(_PHONE).mean() > 0.8 and (as_str.str.len() >= 10).mean() > 0.8:
            return "pii"
        integer_like = bool((numeric.dropna() % 1 == 0).all())
        is_pct = as_str.str.match(_PCT).mean() > 0.5
        if all_unique and integer_like and not is_pct and _is_index(numeric.dropna()):
            return "id"
        return "numeric"
    if _PII_NAME.search(name) and n_unique >= 0.5 * n:
        return "pii"
    if as_str.str.len().mean() > 30 or (as_str.str.contains(" ").mean() > 0.5 and all_unique):
        return "text"
    if all_unique:
        return "id"
    if n_unique > max(MAX_CATEGORIES, 0.05 * n):
        return "text"
    return "categorical"


def _is_index(values: pd.Series) -> bool:
    """Consecutive integers (1..n, 1001..1000+n): a row index or serial number."""
    v = values.sort_values().to_numpy()
    return len(v) > 1 and (v[-1] - v[0]) == len(v) - 1


def _value_counts(s: pd.Series, limit: int = 30) -> list[dict]:
    vc = s.dropna().astype(str).value_counts()
    return [{"value": str(k), "count": int(v)} for k, v in vc.head(limit).items()]


def profile(df: pd.DataFrame) -> dict:
    columns = []
    for name in df.columns:
        s = df[name]
        role = _role(name, s)
        n_unique = int(s.nunique(dropna=True))
        entry = {
            "name": name,
            "role": role,
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": n_unique,
            "samples": [str(v) for v in s.dropna().unique()[:5]],
        }
        if n_unique <= 30:
            entry["values"] = _value_counts(s)
        if role == "numeric":
            num = to_numeric(s)
            entry["stats"] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in
                              {"min": num.min(), "median": num.median(), "max": num.max()}.items()}
        columns.append(entry)

    def target_score(c: dict) -> float:
        if not 2 <= c["n_unique"] <= MAX_TARGET_LEVELS or c["role"] in ("id", "pii", "empty"):
            return -1
        score = 2.0 if _TARGET_NAME.search(c["name"]) else 0.0
        values = [v["value"] for v in c.get("values", [])]
        score += 1.5 if any(_POSITIVE.search(v) for v in values) else 0
        score += 0.25 if c is columns[-1] else 0
        return score

    best = max(columns, key=target_score, default=None)
    target = best["name"] if best is not None and target_score(best) > 0 else None
    positive, unknown, negative = [], [], []
    if target:
        for v in (x["value"] for x in best.get("values", [])):
            if _UNKNOWN.search(v):
                unknown.append(v)
            elif _POSITIVE.search(v):
                positive.append(v)
            else:
                negative.append(v)
    ids = [c["name"] for c in columns if c["role"] == "id"]
    pii_names = [c["name"] for c in columns if c["role"] == "pii" and
                 re.search(r"name", c["name"], re.I)]
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": columns,
        "suggested_target": target,
        "suggested_positive": positive,
        "suggested_negative": negative,
        "suggested_unknown": unknown,
        "suggested_id": (ids or pii_names or [None])[0],
    }
