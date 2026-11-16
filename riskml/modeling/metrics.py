"""Evaluation metrics. The positive class (1) is always "at risk"."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_EPS = 1e-6


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def binary_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "accuracy": _r(accuracy_score(y, pred)),
        "precision": _r(precision_score(y, pred, zero_division=0)),
        "recall": _r(recall_score(y, pred, zero_division=0)),
        "f1": _r(f1_score(y, pred, zero_division=0)),
        "roc_auc": _r(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": _r(average_precision_score(y, p)) if both else float("nan"),
        "mcc": _r(matthews_corrcoef(y, pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y, pred)),
        "brier": _r(brier_score_loss(y, p)),
        "log_loss": _r(log_loss(y, p, labels=[0, 1])),
        "threshold": _r(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 on (y, p). Use on out-of-fold scores only."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, _EPS, None)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]) if len(thresholds) else 0.5


_CI_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc")


def bootstrap_ci(y, p, threshold: float, n: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals over the evaluation set."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in _CI_KEYS}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in _CI_KEYS:
            samples[k].append(m[k])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (_r(np.percentile(v, lo_q)), _r(np.percentile(v, hi_q))) for k, v in
            samples.items()}


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar's test (with continuity correction) on paired classifier decisions."""
    y = np.asarray(y)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    if b01 + b10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "a_only_correct": 0, "b_only_correct": 0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return {"statistic": _r(stat), "p_value": float(chi2.sf(stat, 1)),
            "a_only_correct": b01, "b_only_correct": b10}


def _downsample(points: np.ndarray, n: int) -> list[list[float]]:
    if len(points) > n:
        idx = np.unique(np.linspace(0, len(points) - 1, n).round().astype(int))
        points = points[idx]
    return [[_r(a), _r(b)] for a, b in points]


def curve_points(y, p, n: int = 101) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    roc = _downsample(np.column_stack([fpr, tpr]), n)
    pr = _downsample(np.column_stack([recall, precision])[::-1], n)
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    calib = []
    for b in range(10):
        mask = which == b
        if mask.any():
            calib.append([_r(np.mean(np.asarray(p)[mask])), _r(np.mean(np.asarray(y)[mask])),
                          int(mask.sum())])
    return {"roc": roc, "pr": pr, "calibration": calib}


def finite_or_none(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value
