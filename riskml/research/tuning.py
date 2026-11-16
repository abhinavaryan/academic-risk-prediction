"""Bayesian hyper-parameter optimisation (Optuna TPE) with student-grouped cross-validation."""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from riskml.modeling.zoo import pipeline_for

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _hgb(trial, seed, n_jobs):
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
        max_iter=trial.suggest_int("max_iter", 150, 600),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
        random_state=seed,
    )


def _lgbm(trial, seed, n_jobs):
    from lightgbm import LGBMClassifier  # noqa: PLC0415

    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=seed, n_jobs=n_jobs, verbose=-1,
    )


def _xgb(trial, seed, n_jobs):
    from xgboost import XGBClassifier  # noqa: PLC0415

    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
    )


def _rf(trial, seed, n_jobs):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 150, 400),
        max_depth=trial.suggest_int("max_depth", 6, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        n_jobs=n_jobs, random_state=seed,
    )


SPACES = {
    "Gradient Boosting (HistGB)": _hgb,
    "LightGBM": _lgbm,
    "XGBoost": _xgb,
    "Random Forest": _rf,
}


def tune(name, X, y, splits, numeric, categorical, n_trials, seed, n_jobs=4):
    """Return (best_estimator_unfitted, best_params, history)."""
    build = SPACES[name]

    def objective(trial):
        pipe = pipeline_for(name, build(trial, seed, n_jobs), numeric, categorical)
        scores = []
        for tr, va in splits:
            model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            scores.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = -np.inf
    history = []
    for t in study.trials:
        best = max(best, t.value)
        history.append({"trial": t.number, "auc": round(t.value, 5), "best": round(best, 5)})
    fixed = optuna.trial.FixedTrial(study.best_params)
    return build(fixed, seed, n_jobs), study.best_params, history
