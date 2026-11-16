"""Builds a tiny but real artifacts directory (skops model + schema + manifest) for tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier

from riskml.modeling.zoo import pipeline_for

FEATURES = [
    {
        "name": "clicks",
        "label": "Total LMS clicks",
        "group": "vle",
        "kind": "numeric",
        "integer": True,
        "median": 100.0,
        "p01": 0.0,
        "p99": 900.0,
        "observed_min": 0.0,
        "observed_max": 1000.0,
        "min": 0.0,
        "max": 2000.0,
        "nullable": False,
    },
    {
        "name": "score",
        "label": "Average score",
        "group": "assessment",
        "kind": "numeric",
        "integer": False,
        "median": 60.0,
        "p01": 5.0,
        "p99": 99.0,
        "observed_min": 0.0,
        "observed_max": 100.0,
        "min": 0.0,
        "max": 100.0,
        "nullable": True,
    },
    {
        "name": "gender",
        "label": "Gender",
        "group": "demographic",
        "kind": "categorical",
        "levels": ["F", "M"],
        "mode": "M",
        "nullable": False,
    },
]


def _train():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "clicks": rng.integers(0, 1000, n).astype(float),
            "score": rng.uniform(0, 100, n),
            "gender": rng.choice(["F", "M"], n),
        }
    )
    logit = 2.5 - X["clicks"] / 150 - X["score"] / 40
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    pipe = pipeline_for(
        "Gradient Boosting (HistGB)",
        HistGradientBoostingClassifier(max_iter=60, random_state=0),
        ["clicks", "score"],
        ["gender"],
    )
    return pipe.fit(X, y)


def write_artifacts(root, cutoffs=(0, 60)):
    model = _train()
    files = {}
    for c in cutoffs:
        mp = root / "models" / f"hgb_day{c}.skops"
        mp.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(model, mp)
        schema = {
            "cutoff": c,
            "version": "test",
            "model": "Gradient Boosting (HistGB)",
            "model_file": mp.name,
            "threshold": 0.45,
            "bands": {"moderate": 0.33, "high": 0.66},
            "metrics": {"roc_auc": 0.8, "f1": 0.7},
            "n_train": 600,
            "test_presentation": "2014J",
            "features": FEATURES,
        }
        sp = root / "schema" / f"day{c}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(schema))
        samples = [
            {
                "id": "AAA-2014J-1",
                "final_result": "Pass",
                "at_risk": 0,
                "features": {"clicks": 800.0, "score": 90.0, "gender": "F"},
            },
            {
                "id": "AAA-2014J-2",
                "final_result": "Withdrawn",
                "at_risk": 1,
                "features": {"clicks": 3.0, "score": None, "gender": "M"},
            },
        ]
        smp = root / "samples" / f"day{c}.json"
        smp.parent.mkdir(parents=True, exist_ok=True)
        smp.write_text(json.dumps(samples))
        for p in (mp, sp, smp):
            files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "test",
                "cutoffs": list(cutoffs),
                "primary_cutoff": cutoffs[-1],
                "files": files,
            }
        )
    )
    return root


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    return write_artifacts(tmp_path_factory.mktemp("artifacts"))
