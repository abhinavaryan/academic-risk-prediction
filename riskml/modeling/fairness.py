"""Group-wise performance to audit whether the model treats student groups differently."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_fairness(y, p, threshold: float, groups: pd.Series, min_size: int = 1) -> list[dict]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    g = pd.Series(groups).fillna("Unknown").astype(str).to_numpy()
    rows = []
    for name in sorted(set(g)):
        m = g == name
        n = int(m.sum())
        if n < min_size:
            continue
        yt, pt, pr = y[m], pred[m], p[m]
        pos, neg = yt == 1, yt == 0
        rows.append(
            {
                "group": name,
                "n": n,
                "base_rate": round(float(yt.mean()), 4),
                "positive_rate": round(float(pt.mean()), 4),
                "tpr": round(float(pt[pos].mean()), 4) if pos.any() else None,
                "fpr": round(float(pt[neg].mean()), 4) if neg.any() else None,
                "precision": round(float(yt[pt == 1].mean()), 4) if (pt == 1).any() else None,
                "roc_auc": round(float(roc_auc_score(yt, pr)), 4)
                if pos.any() and neg.any()
                else None,
            }
        )
    return rows


def fairness_gaps(rows: list[dict]) -> dict:
    """Max-min gaps: equal-opportunity (TPR), FPR and demographic-parity (positive rate)."""

    def gap(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(max(vals) - min(vals), 4) if len(vals) >= 2 else None

    return {"tpr_gap": gap("tpr"), "fpr_gap": gap("fpr"), "positive_rate_gap": gap("positive_rate")}
