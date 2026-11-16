import subprocess
import sys

import numpy as np
import pandas as pd

from riskml.modeling.zoo import SERVING_MODELS, make_preprocessor, serving_zoo


def _toy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
        "c": rng.choice(["x", "y", "z"], n),
    })
    y = ((X["a"] + (X["c"] == "x")) > 0.5).astype(int)
    return X, y


def test_preprocessor_handles_missing_and_unknown_categories():
    X, _ = _toy()
    pre = make_preprocessor(["a", "b"], ["c"], scale=True).fit(X)
    out = pre.transform(pd.DataFrame({"a": [np.nan], "b": [np.nan], "c": ["never-seen"]}))
    assert np.isfinite(out).all()


def test_serving_zoo_models_fit_and_predict_proba():
    X, y = _toy()
    zoo = serving_zoo(["a", "b"], ["c"], seed=0)
    assert list(zoo) == list(SERVING_MODELS)
    for name, model in zoo.items():
        p = model.fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (len(X),), name
        assert ((p >= 0) & (p <= 1)).all(), name


def test_serving_zoo_has_no_research_only_imports():
    code = (
        "import sys; from riskml.modeling.zoo import serving_zoo; serving_zoo(['a'], ['c'], 0);"
        "bad=[m for m in ('lightgbm','xgboost','catboost','optuna') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
