"""Explainability helpers: map transformed columns back to raw features; LightGBM TreeSHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def raw_feature_of(transformed: str, categorical: list[str]) -> str:
    """'num__missingindicator_x' -> 'x'; 'cat__gender_M' -> 'gender'; 'num__x' -> 'x'."""
    prefix, _, rest = transformed.partition("__")
    if prefix == "num":
        return rest.removeprefix("missingindicator_")
    for cat in sorted(categorical, key=len, reverse=True):
        if rest == cat or rest.startswith(cat + "_"):
            return cat
    return rest


def lgbm_shap(
    pipeline,
    X: pd.DataFrame,
    categorical: list[str],
    top: int = 12,
    n_points: int = 400,
    seed: int = 42,
) -> dict:
    """Native TreeSHAP of a fitted Pipeline(prep, LGBMClassifier), aggregated to raw features."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Z = prep.transform(X)
    names = prep.get_feature_names_out()
    contrib = model.booster_.predict(Z, pred_contrib=True)[:, :-1]  # last column = bias
    raw = [raw_feature_of(n, categorical) for n in names]
    frame = pd.DataFrame(contrib, columns=names).T.groupby(raw).sum().T
    mean_abs = frame.abs().mean().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_points, len(X)), replace=False)
    beeswarm = []
    for feat in mean_abs.index[:top]:
        col = X[feat]
        if col.dtype == object:
            codes = col.astype("category").cat.codes.replace(-1, np.nan)
            pct = codes.rank(pct=True)
        else:
            pct = col.rank(pct=True)
        beeswarm.append(
            {
                "feature": feat,
                "points": [
                    [
                        round(float(frame[feat].iloc[i]), 4),
                        None if np.isnan(pct.iloc[i]) else round(float(pct.iloc[i]), 3),
                    ]
                    for i in idx
                ],
            }
        )
    return {
        "mean_abs": [{"feature": f, "value": round(float(v), 5)} for f, v in mean_abs.items()],
        "beeswarm": beeswarm,
    }
