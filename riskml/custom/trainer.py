"""Train, compare and apply models on a user's own dataset (e.g. a college's student records).

Workflow
1. Rows whose target value is "at risk" or "not at risk" are *labeled*; rows whose result is
   blank or marked unknown (e.g. "Enrolled") are *scored*.
2. Feature columns are chosen from the profile (IDs, PII, free text, constants are dropped) and
   screened for target leakage (a single column that almost perfectly predicts the target).
3. Every model in the serving zoo is evaluated with stratified k-fold cross-validation inside a
   time budget. The decision threshold is tuned on out-of-fold predictions.
4. The best model (by cross-validated ROC-AUC) is refit on all labeled rows and scores the
   unlabeled rows. Labeled rows keep their honest out-of-fold probability.
5. Each scored student gets up to three reasons (baseline-replacement attribution).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from riskml.config import (
    CUSTOM_TIME_BUDGET_S,
    MAX_REASON_ROWS,
    MIN_CLASS_COUNT,
    MIN_LABELED_ROWS,
    SEED,
)
from riskml.custom.profile import profile, to_numeric
from riskml.modeling.metrics import best_f1_threshold, binary_metrics, curve_points
from riskml.modeling.zoo import MODEL_INFO, SERVING_MODELS, serving_zoo
from riskml.serving.predictor import band_of

LEAKAGE_AUC = 0.98
REASON_ROWS_OVER_BUDGET = 300
MAX_PREDICTION_ROWS = 10_000
_UNUSABLE = {
    "id": "identifier column",
    "pii": "personal data (never used)",
    "text": "free text",
    "constant": "same value in every row",
    "empty": "no values",
}


class TrainingError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class TrainConfig:
    target: str
    positive_values: list[str] = field(default_factory=list)
    negative_values: list[str] | None = None
    unknown_values: list[str] = field(default_factory=list)
    positive_below: float | None = None
    id_column: str | None = None
    exclude: list[str] = field(default_factory=list)
    models: list[str] | None = None


def _norm(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def _labels(df: pd.DataFrame, cfg: TrainConfig, warnings: list[str]) -> pd.Series:
    """1 = at risk, 0 = not at risk, NaN = to be scored, -1 = ignored."""
    raw = df[cfg.target]
    if cfg.positive_below is not None:
        num = to_numeric(raw)
        y = pd.Series(np.where(num < cfg.positive_below, 1.0, 0.0), index=df.index)
        y[num.isna()] = np.nan
        return y
    pos = {_norm(v) for v in cfg.positive_values}
    unknown = {_norm(v) for v in cfg.unknown_values} | {""}
    neg = None if cfg.negative_values is None else {_norm(v) for v in cfg.negative_values}
    values = raw.map(_norm)
    y = pd.Series(np.nan, index=df.index)
    y[values.isin(pos)] = 1.0
    if neg is None:
        y[~values.isin(pos) & ~values.isin(unknown)] = 0.0
    else:
        y[values.isin(neg)] = 0.0
        other = ~values.isin(pos | neg | unknown)
        if other.any():
            warnings.append(
                f"{int(other.sum())} rows have a result value that was not mapped and were ignored."
            )
            y[other] = -1.0
    return y


def _single_feature_auc(x: pd.Series, y: np.ndarray, numeric: bool) -> float:
    if numeric:
        v = x.fillna(x.median())
    else:
        codes = x.fillna("__missing__").astype(str)
        rate = pd.Series(y, index=x.index).groupby(codes).mean()
        v = codes.map(rate)
    if v.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, v)
    return max(auc, 1 - auc)


def _prepare(
    df: pd.DataFrame,
    cfg: TrainConfig,
    prof: dict,
    labeled: pd.Series,
    y: np.ndarray,
    warnings: list[str],
):
    roles = {c["name"]: c["role"] for c in prof["columns"]}
    skip = {cfg.target, cfg.id_column, *cfg.exclude}
    excluded = [{"column": c, "reason": "excluded by you"} for c in cfg.exclude if c in roles]
    numeric, categorical = [], []
    for col, role in roles.items():
        if col in skip:
            continue
        if role in _UNUSABLE:
            excluded.append({"column": col, "reason": _UNUSABLE[role]})
        elif role == "numeric":
            numeric.append(col)
        else:
            categorical.append(col)
    X = pd.DataFrame(index=df.index)
    for c in numeric:
        X[c] = to_numeric(df[c]).astype("float64")
    for c in categorical:
        X[c] = df[c].map(lambda v: v if isinstance(v, str) else (np.nan if pd.isna(v) else str(v)))
    for c in [*numeric, *categorical]:
        auc = _single_feature_auc(X.loc[labeled, c], y, c in numeric)
        if auc >= LEAKAGE_AUC:
            excluded.append(
                {"column": c, "reason": f"possible target leak (AUC {auc:.3f} on its own)"}
            )
            warnings.append(
                f"'{c}' predicts the result almost perfectly on its own "
                f"(AUC {auc:.3f}); it looks like it leaks the answer and was "
                "excluded."
            )
    leaked = {e["column"] for e in excluded}
    numeric = [c for c in numeric if c not in leaked]
    categorical = [c for c in categorical if c not in leaked]
    if not numeric and not categorical:
        raise TrainingError(
            "no_features",
            "No usable feature columns are left after removing "
            "IDs, personal data, text and leaky columns.",
        )
    return X[numeric + categorical], numeric, categorical, excluded


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _reasons(
    model, X: pd.DataFrame, baseline: dict, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline-replacement attribution for every row and feature.

    Returns (probability impacts, log-odds impacts). Log-odds stay informative when a risk is
    saturated near 0% or 100%, where single replacements barely move the probability.
    """
    impacts = np.zeros((len(X), X.shape[1]))
    log_odds = np.zeros_like(impacts)
    for j, col in enumerate(X.columns):
        Xb = X.copy()
        Xb[col] = baseline[col]
        pb = model.predict_proba(Xb)[:, 1]
        impacts[:, j] = p - pb
        log_odds[:, j] = _logit(p) - _logit(pb)
    return impacts, log_odds


def _fmt(v):
    if isinstance(v, float):
        return None if np.isnan(v) else round(v, 3)
    return v


def train_custom(
    df: pd.DataFrame,
    cfg: TrainConfig,
    time_budget_s: float = CUSTOM_TIME_BUDGET_S,
    seed: int = SEED,
) -> dict:
    started = time.perf_counter()
    warnings: list[str] = []
    for col in [cfg.target, cfg.id_column, *cfg.exclude]:
        if col is not None and col not in df.columns:
            raise TrainingError("bad_config", f"Column '{col}' is not in the file.")
    if cfg.positive_below is None and not cfg.positive_values:
        raise TrainingError("bad_config", "Choose which result values mean 'at risk'.")
    unknown_models = set(cfg.models or []) - set(SERVING_MODELS)
    if unknown_models:
        raise TrainingError("bad_config", f"Unknown models: {sorted(unknown_models)}")

    df = df.reset_index(drop=True)
    y_all = _labels(df, cfg, warnings)
    labeled = y_all.isin([0.0, 1.0])
    scored = y_all.isna()
    y = y_all[labeled].astype(int).to_numpy()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if len(y) < MIN_LABELED_ROWS:
        raise TrainingError(
            "too_few_rows",
            f"Need at least {MIN_LABELED_ROWS} rows with a known result; found {len(y)}.",
        )
    if min(n_pos, n_neg) == 0:
        raise TrainingError(
            "single_class",
            "All labeled rows have the same result, so there is "
            "nothing to learn. Check the at-risk values.",
        )
    if min(n_pos, n_neg) < MIN_CLASS_COUNT:
        raise TrainingError(
            "too_few_rows",
            f"Need at least {MIN_CLASS_COUNT} at-risk and {MIN_CLASS_COUNT} not-at-risk examples.",
        )

    prof = profile(df)
    X, numeric, categorical, excluded = _prepare(df, cfg, prof, labeled, y, warnings)
    X_lab = X[labeled].reset_index(drop=True)
    n_splits = int(min(5, n_pos, n_neg))
    splits = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X_lab, y)
    )
    zoo = serving_zoo(
        numeric, categorical, seed=seed, n_samples=len(y) * (n_splits - 1) // n_splits
    )
    names = [m for m in SERVING_MODELS if cfg.models is None or m in cfg.models]

    leaderboard, oofs, skipped = [], {}, []
    for i, name in enumerate(names):
        if i > 0 and time.perf_counter() - started > time_budget_s:
            skipped.append(name)
            continue
        t0 = time.perf_counter()
        oof = np.zeros(len(y))
        aucs = []
        over_budget = False
        for tr, va in splits:
            model = clone(zoo[name]).fit(X_lab.iloc[tr], y[tr])
            oof[va] = model.predict_proba(X_lab.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y[va], oof[va]) if len(set(y[va])) == 2 else np.nan)
            if i > 0 and time.perf_counter() - started > time_budget_s * 1.25:
                over_budget = True
                break
        if over_budget:
            skipped.append(name)
            continue
        thr = best_f1_threshold(y, oof)
        m = binary_metrics(y, oof, thr)
        oofs[name] = (oof, thr)
        leaderboard.append(
            {
                "name": name,
                "family": MODEL_INFO[name]["family"],
                **{
                    k: m[k]
                    for k in (
                        "roc_auc",
                        "f1",
                        "precision",
                        "recall",
                        "accuracy",
                        "pr_auc",
                        "mcc",
                        "balanced_accuracy",
                    )
                },
                "roc_auc_std": round(float(np.nanstd(aucs)), 4),
                "threshold": round(thr, 4),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
    if skipped:
        warnings.append(
            f"Skipped {', '.join(skipped)} to stay within the {time_budget_s:g}s time budget."
        )
    leaderboard.sort(key=lambda r: (r["roc_auc"], r["f1"]), reverse=True)
    champion = leaderboard[0]["name"]
    oof, thr = oofs[champion]
    final = clone(zoo[champion]).fit(X_lab, y)

    proba = pd.Series(np.nan, index=df.index)
    proba[labeled] = oof
    X_score = X[scored]
    if len(X_score):
        proba[scored] = final.predict_proba(X_score)[:, 1]

    # Explain: every scored row, then the riskiest labeled rows, up to MAX_REASON_ROWS.
    order = list(df.index[scored]) + list(proba[labeled].sort_values(ascending=False).index)
    over_budget = time.perf_counter() - started > time_budget_s
    explain_idx = order[: REASON_ROWS_OVER_BUDGET if over_budget else MAX_REASON_ROWS]
    if over_budget and len(order) > len(explain_idx):
        warnings.append(
            f"Reasons were computed for the {len(explain_idx)} highest-priority rows "
            "only, to stay within the time budget."
        )
    baseline = {
        c: (
            X_lab[c].median()
            if c in numeric
            else (X_lab[c].mode().iloc[0] if X_lab[c].notna().any() else np.nan)
        )
        for c in X.columns
    }
    reasons_by_row: dict[int, list] = {}
    importance = []
    if explain_idx:
        Xe = X.loc[explain_idx]
        pe = final.predict_proba(Xe)[:, 1]
        impacts, log_odds = _reasons(final, Xe, baseline, pe)
        cols = list(X.columns)
        for row_i, idx in enumerate(explain_idx):
            top = [j for j in np.argsort(-log_odds[row_i])[:3] if log_odds[row_i, j] > 0.05]
            reasons_by_row[idx] = [
                {
                    "feature": cols[j],
                    "value": _fmt(Xe.iloc[row_i, j]),
                    "typical": _fmt(baseline[cols[j]]),
                    "impact": round(max(0.0, float(impacts[row_i, j])), 4),
                    "log_odds": round(float(log_odds[row_i, j]), 4),
                }
                for j in top
            ]
        mean_abs = np.abs(impacts).mean(axis=0)
        importance = sorted(
            (
                {"feature": c, "mean_abs_impact": round(float(v), 4)}
                for c, v in zip(cols, mean_abs, strict=True)
            ),
            key=lambda r: r["mean_abs_impact"],
            reverse=True,
        )

    ids = df[cfg.id_column] if cfg.id_column else None
    predictions = []
    out_order = list(df.index[scored]) + list(proba[labeled].sort_values(ascending=False).index)
    if len(out_order) > MAX_PREDICTION_ROWS:
        warnings.append(
            f"Showing the {MAX_PREDICTION_ROWS:,} highest-priority rows of {len(out_order):,}."
        )
        out_order = out_order[:MAX_PREDICTION_ROWS]
    for idx in out_order:
        p = float(proba[idx])
        is_scored = bool(scored[idx])
        label = _norm(ids[idx]) if ids is not None else ""
        predictions.append(
            {
                "row": int(idx) + 2,  # spreadsheet row number (row 1 is the header)
                "label": label or f"Row {int(idx) + 2}",
                "probability": round(p, 4),
                "band": band_of(p),
                "at_risk": p >= thr,
                "actual": None if is_scored else int(y_all[idx]),
                "source": "scored" if is_scored else "cross-validated",
                "reasons": reasons_by_row.get(idx, []),
            }
        )

    champ_metrics = binary_metrics(y, oof, thr)
    return {
        "summary": {
            "n_rows": len(df),
            "n_labeled": len(y),
            "n_scored": int(scored.sum()),
            "n_ignored": int((y_all == -1).sum()),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": round(n_pos / len(y), 4),
            "cv_folds": n_splits,
            "features_used": list(X.columns),
            "excluded": excluded,
            "target": cfg.target,
            "positive_values": cfg.positive_values,
            "negative_values": cfg.negative_values,
            "unknown_values": cfg.unknown_values,
            "positive_below": cfg.positive_below,
            "id_column": cfg.id_column,
            "n_explained": len(explain_idx),
        },
        "leaderboard": leaderboard,
        "champion": champion,
        "threshold": round(thr, 4),
        "confusion": {k: champ_metrics[k] for k in ("tp", "fp", "tn", "fn")},
        "curves": curve_points(y, oof),
        "importance": importance,
        "predictions": predictions,
        "warnings": warnings,
        "seconds": round(time.perf_counter() - started, 2),
    }
