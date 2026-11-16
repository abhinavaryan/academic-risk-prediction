"""Model zoo.

* ``research_zoo`` - every algorithm compared in the research study (includes LightGBM, XGBoost
  and CatBoost, imported lazily so the web API never loads them).
* ``serving_zoo``  - the scikit-learn-only subset used for bring-your-own-data training inside the
  web API (small install footprint, no system OpenMP requirement).

All models share one preprocessing recipe: median imputation with missing-value indicators for
numeric columns, constant imputation + one-hot encoding (rare levels pooled) for categoricals.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

MODEL_INFO: dict[str, dict[str, str]] = {
    "Baseline (majority)": {"family": "Baseline", "short": "Baseline"},
    "Logistic Regression": {"family": "Linear", "short": "LogReg"},
    "Naive Bayes": {"family": "Probabilistic", "short": "NB"},
    "k-Nearest Neighbours": {"family": "Instance-based", "short": "kNN"},
    "Decision Tree": {"family": "Tree", "short": "DT"},
    "Random Forest": {"family": "Bagging ensemble", "short": "RF"},
    "Extra Trees": {"family": "Bagging ensemble", "short": "ET"},
    "AdaBoost": {"family": "Boosting ensemble", "short": "Ada"},
    "Gradient Boosting (HistGB)": {"family": "Boosting ensemble", "short": "HGB"},
    "Linear SVM": {"family": "Kernel / margin", "short": "SVM"},
    "Neural Network (MLP)": {"family": "Neural network", "short": "MLP"},
    "LightGBM": {"family": "Boosting ensemble", "short": "LGBM"},
    "XGBoost": {"family": "Boosting ensemble", "short": "XGB"},
    "CatBoost": {"family": "Boosting ensemble", "short": "CatB"},
    "Stacked Ensemble": {"family": "Stacking ensemble", "short": "Stack"},
}

# Ordered fastest/strongest first so a tight time budget still yields good candidates.
SERVING_MODELS = (
    "Logistic Regression",
    "Gradient Boosting (HistGB)",
    "Random Forest",
    "Extra Trees",
    "Decision Tree",
    "Naive Bayes",
    "k-Nearest Neighbours",
    "Neural Network (MLP)",
)

# Models that need scaled inputs; tree ensembles do not.
_SCALED = {"Logistic Regression", "Naive Bayes", "k-Nearest Neighbours", "Linear SVM",
           "Neural Network (MLP)"}


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list = [(
        "impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    )]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), list(numeric)))
    if categorical:
        transformers.append(("cat", cat, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop")


def _pipe(name: str, estimator, numeric, categorical) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical, scale=name in _SCALED)),
        ("model", estimator),
    ])


def _knn_k(n_samples: int | None) -> int:
    if n_samples is None:
        return 25
    return int(max(1, min(25, n_samples // 10)))


def _sklearn_estimators(seed: int, n_jobs: int, n_samples: int | None) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=n_jobs, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=n_jobs, random_state=seed),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=seed),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=_knn_k(n_samples),
                                                     weights="distance", n_jobs=n_jobs),
        "Naive Bayes": GaussianNB(),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                              early_stopping=True, max_iter=300,
                                              random_state=seed),
    }


def serving_zoo(numeric: list[str], categorical: list[str], seed: int, n_jobs: int = 2,
                n_samples: int | None = None) -> dict[str, Pipeline]:
    est = _sklearn_estimators(seed, n_jobs, n_samples)
    return {name: _pipe(name, est[name], numeric, categorical) for name in SERVING_MODELS}


def research_estimators(seed: int, n_jobs: int = 4) -> dict:
    """All research estimators (unfitted). Third-party boosters are imported here only."""
    from catboost import CatBoostClassifier  # noqa: PLC0415
    from lightgbm import LGBMClassifier  # noqa: PLC0415
    from xgboost import XGBClassifier  # noqa: PLC0415

    est = _sklearn_estimators(seed, n_jobs, None)
    return {
        "Baseline (majority)": DummyClassifier(strategy="prior"),
        "Logistic Regression": est["Logistic Regression"],
        "Naive Bayes": est["Naive Bayes"],
        "k-Nearest Neighbours": est["k-Nearest Neighbours"],
        "Decision Tree": est["Decision Tree"],
        "Random Forest": est["Random Forest"],
        "Extra Trees": est["Extra Trees"],
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed),
        "Gradient Boosting (HistGB)": est["Gradient Boosting (HistGB)"],
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000, random_state=seed),
                                             cv=3),
        "Neural Network (MLP)": est["Neural Network (MLP)"],
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=seed, n_jobs=n_jobs, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 eval_metric="logloss", random_state=seed, n_jobs=n_jobs),
        "CatBoost": CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, verbose=0,
                                       random_seed=seed, thread_count=n_jobs),
    }


def research_zoo(numeric: list[str], categorical: list[str], seed: int,
                 n_jobs: int = 4) -> dict[str, Pipeline]:
    return {name: _pipe(name, est, numeric, categorical)
            for name, est in research_estimators(seed, n_jobs).items()}


def pipeline_for(name: str, estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Wrap a (possibly tuned) estimator in the shared preprocessing."""
    return _pipe(name, estimator, numeric, categorical)
