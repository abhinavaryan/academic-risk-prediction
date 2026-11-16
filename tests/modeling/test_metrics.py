import numpy as np
import pandas as pd
import pytest

from riskml.modeling.fairness import group_fairness
from riskml.modeling.metrics import (
    best_f1_threshold,
    binary_metrics,
    bootstrap_ci,
    curve_points,
    mcnemar,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def test_perfect_predictions_metrics():
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    m = binary_metrics(Y, p, threshold=0.5)
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (4, 0, 4, 0)
    assert m["threshold"] == 0.5


def test_metrics_keys_complete():
    m = binary_metrics(Y, np.linspace(0.1, 0.9, 8), 0.5)
    expected = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc",
                "balanced_accuracy", "brier", "log_loss", "threshold", "tp", "fp", "tn", "fn"}
    assert set(m) == expected


def test_best_threshold_beats_half_on_skewed_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.15 + 0.2 * y + rng.normal(0, 0.08, 2000), 0, 1)  # all scores below ~0.5
    t = best_f1_threshold(y, p)
    assert binary_metrics(y, p, t)["f1"] > binary_metrics(y, p, 0.5)["f1"]
    assert 0 < t < 0.5


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 500), 0, 1)
    ci = bootstrap_ci(y, p, threshold=0.5, n=200, seed=0)
    point = binary_metrics(y, p, 0.5)
    for k in ("f1", "roc_auc", "accuracy", "recall", "precision"):
        lo, hi = ci[k]
        assert lo <= point[k] <= hi


def test_mcnemar_identical_preds_p_is_1():
    pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    assert mcnemar(Y, pred, pred)["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    y = np.array([1] * 100)
    good = np.ones(100, dtype=int)
    bad = np.zeros(100, dtype=int)
    assert mcnemar(y, good, bad)["p_value"] < 1e-6


def test_curve_points_shape_and_bounds():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    c = curve_points(y, p, n=51)
    assert len(c["roc"]) <= 51 + 2 and c["roc"][0] == [0.0, 0.0] and c["roc"][-1] == [1.0, 1.0]
    assert all(0 <= r <= 1 and 0 <= q <= 1 for r, q in c["pr"])
    assert sum(b[2] for b in c["calibration"]) == 300


def test_group_fairness_rows():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])
    g = pd.Series(["F", "F", "F", "M", "M", "M"])
    rows = {r["group"]: r for r in group_fairness(y, p, 0.5, g)}
    assert rows["F"]["n"] == 3 and rows["F"]["tpr"] == pytest.approx(0.5)
    assert rows["M"]["fpr"] == pytest.approx(0.5)
    assert rows["M"]["positive_rate"] == pytest.approx(2 / 3, abs=1e-4)  # 4-dp rounding
