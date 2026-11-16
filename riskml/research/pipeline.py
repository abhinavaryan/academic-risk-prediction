"""End-to-end research pipeline on the full OULAD dataset.

    python -m riskml.research.pipeline --data-dir data/oulad          # full study (~1 h, 4 cores)
    python -m riskml.research.pipeline --data-dir data/oulad --quick  # smoke run (~2 min)

What it does (see docs/METHODOLOGY.md for the reasoning behind each step):

1. Exploratory statistics of the raw tables (counts, outcome mix, engagement curves).
2. Primary study at day 60: every model in the zoo, 5-fold student-grouped CV on the
   2013B/2013J/2014B presentations, Optuna tuning of four boosters/forests, a stacked ensemble,
   then a single evaluation on the unseen 2014J presentation with bootstrap CIs and McNemar tests.
3. Early-warning timeline: the same study at days 0..180 for four representative models.
4. Feature-group ablation, permutation importance, TreeSHAP, and a fairness audit.
5. Writes served artifacts (artifacts/), dashboard data (web/src/generated/), figures and a
   Markdown results report (reports/).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import skops
import skops.io as sio
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from riskml import __version__
from riskml.config import (
    BAND_HIGH,
    BAND_MODERATE,
    CUTOFFS,
    DEFAULT_DATA_DIR,
    OULAD_SHA256,
    PRIMARY_CUTOFF,
    ROOT,
    SEED,
    TEST_PRESENTATION,
)
from riskml.modeling.fairness import fairness_gaps, group_fairness
from riskml.modeling.metrics import (
    best_f1_threshold,
    binary_metrics,
    bootstrap_ci,
    curve_points,
    mcnemar,
)
from riskml.modeling.zoo import MODEL_INFO, pipeline_for
from riskml.oulad.features import FEATURE_GROUPS, FEATURE_LABELS, FeatureFrame, build_features
from riskml.oulad.load import KEYS, OuladTables, load_tables

DEPLOY_MODEL = "Gradient Boosting (HistGB)"
TUNED = ("Gradient Boosting (HistGB)", "LightGBM", "XGBoost", "Random Forest")
TIMELINE_MODELS = ("Logistic Regression", "Random Forest", "Gradient Boosting (HistGB)",
                   "LightGBM")
STACK_BASES = ("LightGBM (tuned)", "XGBoost (tuned)", "CatBoost",
               "Gradient Boosting (HistGB) (tuned)", "Random Forest (tuned)",
               "Logistic Regression")
PROTECTED = ["gender", "disability", "age_band", "imd_band", "region"]
FAIRNESS_ATTRS = ["gender", "disability", "age_band", "imd_band", "highest_education"]
OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"]
HARD_BOUNDS = {
    "assess_mean_score": (0, 100), "assess_weighted_score": (0, 100),
    "assess_min_score": (0, 100), "assess_tma_mean": (0, 100), "assess_cma_mean": (0, 100),
    "assess_submit_rate": (0, 1), "assess_score_rel_cohort": (-100, 100),
}


@dataclass
class Settings:
    quick: bool = False
    n_folds: int = 5
    n_trials: int = 30
    rf_trials: int = 15
    n_bootstrap: int = 1000
    perm_repeats: int = 5
    cutoffs: tuple[int, ...] = CUTOFFS
    n_jobs: int = 4
    samples_per_cutoff: int = 150
    timeline_models: tuple[str, ...] = TIMELINE_MODELS
    notes: list[str] = field(default_factory=list)

    @classmethod
    def quick_mode(cls) -> Settings:
        return cls(quick=True, n_folds=2, n_trials=0, rf_trials=0, n_bootstrap=50,
                   perm_repeats=1, cutoffs=(0, PRIMARY_CUTOFF), samples_per_cutoff=40,
                   timeline_models=("Logistic Regression", "Gradient Boosting (HistGB)"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _clean(obj):
    """Make an object strictly JSON-serialisable (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- data helpers
def split(ff: FeatureFrame):
    test = (ff.meta["code_presentation"] == TEST_PRESENTATION).to_numpy()
    tr, te = ~test, test
    return (ff.X[tr].reset_index(drop=True), ff.y[tr].reset_index(drop=True),
            ff.groups[tr].reset_index(drop=True), ff.X[te].reset_index(drop=True),
            ff.y[te].reset_index(drop=True), ff.meta[te].reset_index(drop=True))


def cv_splits(X, y, groups, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
                .split(X, y, groups))


def oof_predict(pipe, X, y, splits):
    oof = np.full(len(y), np.nan)
    for tr, va in splits:
        model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


def evaluate(name: str, pipe, Xtr, ytr, Xte, yte, splits, s: Settings, *, ci: bool = True):
    t0 = time.perf_counter()
    oof = oof_predict(pipe, Xtr, ytr, splits)
    cv_seconds = time.perf_counter() - t0
    thr = best_f1_threshold(ytr, oof)
    folds = [binary_metrics(ytr.iloc[va], oof[va], thr) for _, va in splits]
    t0 = time.perf_counter()
    model = clone(pipe).fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = model.predict_proba(Xte)[:, 1]
    predict_ms_per_1k = (time.perf_counter() - t0) * 1000 * 1000 / max(len(Xte), 1)
    cv = binary_metrics(ytr, oof, thr)
    record = {
        "name": name,
        "family": MODEL_INFO.get(name.replace(" (tuned)", ""), {}).get("family", "Ensemble"),
        "tuned": name.endswith("(tuned)"),
        "threshold": round(thr, 4),
        "cv": {
            **{k: cv[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                  "mcc", "balanced_accuracy", "brier")},
            "roc_auc_std": round(float(np.std([f["roc_auc"] for f in folds])), 4),
            "f1_std": round(float(np.std([f["f1"] for f in folds])), 4),
            "folds": [{"roc_auc": f["roc_auc"], "f1": f["f1"]} for f in folds],
        },
        "test": binary_metrics(yte, p, thr),
        "test_at_0_5": binary_metrics(yte, p, 0.5),
        "ci95": bootstrap_ci(yte, p, thr, n=s.n_bootstrap, seed=SEED) if ci else None,
        "timing": {"cv_seconds": round(cv_seconds, 2), "fit_seconds": round(fit_seconds, 2),
                   "predict_ms_per_1k": round(predict_ms_per_1k, 3)},
    }
    return record, model, p, oof


# --------------------------------------------------------------------------- EDA
def eda(t: OuladTables) -> tuple[dict, dict]:
    info = t.info
    sv = t.student_vle
    dataset = {
        "registrations": len(info),
        "students": int(info["id_student"].nunique()),
        "modules": int(info["code_module"].nunique()),
        "presentations": int(info["code_presentation"].nunique()),
        "module_presentations": len(t.courses),
        "vle_interactions": len(sv),
        "vle_clicks": int(sv["sum_click"].sum()),
        "vle_resources": len(t.vle),
        "assessments": len(t.assessments),
        "assessment_submissions": len(t.student_assessment),
    }
    outcome_counts = info["final_result"].value_counts()
    at_risk = info["final_result"].isin(["Fail", "Withdrawn"])

    def rate_by(col: str) -> list[dict]:
        g = info.assign(at_risk=at_risk).groupby(info[col].fillna("Unknown"))["at_risk"]
        return [{"group": str(k), "n": int(v.size), "at_risk_rate": round(float(v.mean()), 4)}
                for k, v in g]

    # Average weekly clicks per registered student, by final outcome.
    reg_out = info[[*KEYS, "final_result"]]
    weekly = sv.assign(week=sv["date"] // 7).merge(reg_out, on=KEYS, how="left")
    clicks = weekly.groupby(["final_result", "week"])["sum_click"].sum().unstack(0).fillna(0)
    n_by_outcome = info["final_result"].value_counts()
    engagement = [
        {"week": int(w), **{o: round(float(clicks.loc[w, o] / n_by_outcome[o]), 2)
                            for o in OUTCOMES if o in clicks.columns}}
        for w in clicks.index if -4 <= w <= 38
    ]
    # How quickly do withdrawals happen? cumulative share of all withdrawals by day.
    wd = t.registration.merge(info, on=KEYS)
    wd = wd.loc[wd["final_result"] == "Withdrawn", "date_unregistration"].dropna()
    withdrawal_curve = [{"day": d, "share": round(float((wd <= d).mean()), 4)}
                        for d in range(-60, 271, 10)]
    # Score distribution by outcome (non-exam assessments over the whole course).
    a = t.assessments.loc[t.assessments["assessment_type"] != "Exam", ["id_assessment"]]
    sa = t.student_assessment.merge(a, on="id_assessment").merge(
        t.assessments[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    mean_score = sa.groupby(KEYS)["score"].mean().rename("mean_score").reset_index()
    ms = mean_score.merge(reg_out, on=KEYS)
    score_box = []
    for o in OUTCOMES:
        v = ms.loc[ms["final_result"] == o, "mean_score"].dropna()
        q = np.percentile(v, [10, 25, 50, 75, 90]) if len(v) else [np.nan] * 5
        score_box.append({"outcome": o, "n": len(v), "p10": q[0], "p25": q[1], "median": q[2],
                          "p75": q[3], "p90": q[4]})
    eda_out = {
        "outcomes": [{"outcome": o, "n": int(outcome_counts.get(o, 0))} for o in OUTCOMES],
        "at_risk_rate": round(float(at_risk.mean()), 4),
        "by_module": rate_by("code_module"),
        "by_imd_band": rate_by("imd_band"),
        "by_highest_education": rate_by("highest_education"),
        "by_age_band": rate_by("age_band"),
        "by_gender": rate_by("gender"),
        "by_disability": rate_by("disability"),
        "engagement_by_week": engagement,
        "withdrawal_curve": withdrawal_curve,
        "score_by_outcome": score_box,
        "presentations": [{"code_module": r.code_module, "code_presentation": r.code_presentation,
                           "length_days": int(r.module_presentation_length),
                           "registrations": int(((info["code_module"] == r.code_module) &
                                                 (info["code_presentation"] ==
                                                  r.code_presentation)).sum())}
                          for r in t.courses.itertuples()],
    }
    return dataset, eda_out


# --------------------------------------------------------------------------- model zoo
def quick_estimators() -> dict:
    return {
        "Baseline (majority)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=50, min_samples_leaf=2,
                                                n_jobs=2, random_state=SEED),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(max_iter=150,
                                                                     random_state=SEED),
    }


def primary_study(ff: FeatureFrame, s: Settings) -> dict:
    Xtr, ytr, gtr, Xte, yte, meta_te = split(ff)
    splits = cv_splits(Xtr, ytr, gtr, s.n_folds)
    num, cat = ff.numeric, ff.categorical
    log(f"primary day {ff.cutoff}: train={len(Xtr)} test={len(Xte)} "
        f"at-risk train={ytr.mean():.3f} test={yte.mean():.3f}")

    if s.quick:
        estimators = quick_estimators()
    else:
        from riskml.modeling.zoo import research_estimators  # noqa: PLC0415

        estimators = research_estimators(SEED, s.n_jobs)

    records, models, test_p, oofs = {}, {}, {}, {}

    def add(name, est):
        pipe = pipeline_for(name.replace(" (tuned)", ""), est, num, cat)
        rec, model, p, oof = evaluate(name, pipe, Xtr, ytr, Xte, yte, splits, s)
        records[name], models[name], test_p[name], oofs[name] = rec, model, p, oof
        log(f"  {name:38s} cv_auc={rec['cv']['roc_auc']:.4f} test_auc={rec['test']['roc_auc']:.4f}"
            f" test_f1={rec['test']['f1']:.4f} ({rec['timing']['cv_seconds']:.0f}s)")

    for name, est in estimators.items():
        add(name, est)

    tuning = {}
    best_params = {}
    if not s.quick:
        from riskml.research.tuning import tune  # noqa: PLC0415

        tune_splits = cv_splits(Xtr, ytr, gtr, 3)
        for name in TUNED:
            trials = s.rf_trials if name == "Random Forest" else s.n_trials
            log(f"  tuning {name} ({trials} Optuna trials, 3-fold grouped CV)")
            est, params, history = tune(name, Xtr, ytr, tune_splits, num, cat, trials, SEED,
                                        s.n_jobs)
            tuning[name] = {"best_params": params, "history": history, "n_trials": trials}
            best_params[name] = params
            add(f"{name} (tuned)", est)

        # Stacked ensemble: logistic meta-learner on out-of-fold probabilities of strong bases.
        bases = [b for b in STACK_BASES if b in oofs]
        meta_tr = np.column_stack([oofs[b] for b in bases])
        meta_te = np.column_stack([test_p[b] for b in bases])
        meta = LogisticRegression(max_iter=1000)
        meta_oof = np.full(len(ytr), np.nan)
        for tr, va in splits:
            meta_oof[va] = clone(meta).fit(meta_tr[tr], ytr.iloc[tr]).predict_proba(
                meta_tr[va])[:, 1]
        thr = best_f1_threshold(ytr, meta_oof)
        p = clone(meta).fit(meta_tr, ytr).predict_proba(meta_te)[:, 1]
        cv = binary_metrics(ytr, meta_oof, thr)
        folds = [binary_metrics(ytr.iloc[va], meta_oof[va], thr) for _, va in splits]
        records["Stacked Ensemble"] = {
            "name": "Stacked Ensemble", "family": "Stacking ensemble", "tuned": False,
            "threshold": round(thr, 4), "bases": bases,
            "cv": {**{k: cv[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                                         "pr_auc", "mcc", "balanced_accuracy", "brier")},
                   "roc_auc_std": round(float(np.std([f["roc_auc"] for f in folds])), 4),
                   "f1_std": round(float(np.std([f["f1"] for f in folds])), 4),
                   "folds": [{"roc_auc": f["roc_auc"], "f1": f["f1"]} for f in folds]},
            "test": binary_metrics(yte, p, thr), "test_at_0_5": binary_metrics(yte, p, 0.5),
            "ci95": bootstrap_ci(yte, p, thr, n=s.n_bootstrap, seed=SEED),
            "timing": {"cv_seconds": None, "fit_seconds": None, "predict_ms_per_1k": None},
        }
        test_p["Stacked Ensemble"] = p
        log(f"  Stacked Ensemble cv_auc={cv['roc_auc']:.4f} "
            f"test_auc={records['Stacked Ensemble']['test']['roc_auc']:.4f}")

    ranked = sorted(records.values(), key=lambda r: (r["cv"]["roc_auc"], r["cv"]["f1"]),
                    reverse=True)
    champion = ranked[0]["name"]
    deploy_name = f"{DEPLOY_MODEL} (tuned)" if f"{DEPLOY_MODEL} (tuned)" in records \
        else DEPLOY_MODEL
    champ_pred = (test_p[champion] >= records[champion]["threshold"]).astype(int)
    for r in ranked:
        pred = (test_p[r["name"]] >= r["threshold"]).astype(int)
        r["mcnemar_vs_champion"] = None if r["name"] == champion else mcnemar(
            yte.to_numpy(), champ_pred, pred)
        r["rank"] = ranked.index(r) + 1
    curves = {name: curve_points(yte, p) for name, p in test_p.items()}
    return {
        "records": ranked, "champion": champion, "deployed": deploy_name, "curves": curves,
        "models": models, "test_p": test_p, "tuning": tuning, "best_params": best_params,
        "data": (Xtr, ytr, gtr, Xte, yte, meta_te, splits),
    }


def deploy_estimator(best_params: dict):
    params = best_params.get(DEPLOY_MODEL)
    if params:
        return HistGradientBoostingClassifier(**params, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=SEED)


def timeline_estimators(best_params: dict, s: Settings) -> dict:
    out = {}
    for name in s.timeline_models:
        if name == "Logistic Regression":
            out[name] = LogisticRegression(max_iter=2000)
        elif name == "Gradient Boosting (HistGB)":
            out[name] = deploy_estimator(best_params)
        elif name == "Random Forest":
            p = best_params.get(name)
            out[name] = (RandomForestClassifier(**p, n_jobs=s.n_jobs, random_state=SEED) if p
                         else RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                     n_jobs=s.n_jobs, random_state=SEED))
        elif name == "LightGBM":
            from lightgbm import LGBMClassifier  # noqa: PLC0415

            p = best_params.get(name, {})
            out[name] = LGBMClassifier(**p, subsample_freq=1, random_state=SEED,
                                       n_jobs=s.n_jobs, verbose=-1)
    return out


# --------------------------------------------------------------------------- artifacts
def feature_schema(ff: FeatureFrame, Xtr: pd.DataFrame) -> list[dict]:
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    feats = []
    for c in ff.X.columns:
        entry = {"name": c, "label": FEATURE_LABELS.get(c, c), "group": group_of[c]}
        if c in ff.categorical:
            levels = sorted(Xtr[c].dropna().astype(str).unique().tolist())
            mode = Xtr[c].mode(dropna=True)
            entry |= {"kind": "categorical", "levels": levels,
                      "mode": str(mode.iloc[0]) if len(mode) else None,
                      "nullable": bool(ff.X[c].isna().any())}
        else:
            col = ff.X[c]
            lo, hi = float(col.min()), float(col.max())
            span = max(hi - lo, 1.0)
            if c in HARD_BOUNDS:
                vlo, vhi = HARD_BOUNDS[c]
            else:
                vlo = 0.0 if lo >= 0 else lo - span
                vhi = hi + span
            entry |= {
                "kind": "numeric",
                "integer": bool(np.all(np.mod(col.dropna(), 1) == 0)),
                "median": float(Xtr[c].median()) if Xtr[c].notna().any() else None,
                "p01": float(col.quantile(0.01)), "p99": float(col.quantile(0.99)),
                "observed_min": lo, "observed_max": hi, "min": vlo, "max": vhi,
                "nullable": bool(col.isna().any()),
            }
        feats.append(entry)
    return feats


def sample_students(ff: FeatureFrame, n: int) -> list[dict]:
    test = ff.meta["code_presentation"] == TEST_PRESENTATION
    meta = ff.meta[test]
    picked = (meta.groupby("final_result", group_keys=False)
              .apply(lambda g: g.sample(n=max(1, round(n * len(g) / len(meta))),
                                        random_state=SEED), include_groups=False))
    idx = picked.index
    out = []
    for i in idx:
        row = ff.X.loc[i]
        feats = {}
        for c, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                feats[c] = None
            elif c in ff.categorical:
                feats[c] = str(v)
            else:
                feats[c] = float(v)
        m = ff.meta.loc[i]
        out.append({"id": f"{m['code_module']}-{m['code_presentation']}-{int(m['id_student'])}",
                    "final_result": m["final_result"], "at_risk": int(ff.y.loc[i]),
                    "features": feats})
    return sorted(out, key=lambda r: r["id"])


def save_deployed(art: Path, cutoff: int, model, ff: FeatureFrame, Xtr, threshold: float,
                  metrics: dict, n_train: int, s: Settings) -> list[Path]:
    model_path = art / "models" / f"hgb_day{cutoff}.skops"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, model_path)
    schema = {
        "cutoff": cutoff, "version": __version__, "model": DEPLOY_MODEL,
        "model_file": model_path.name, "threshold": round(threshold, 4),
        "bands": {"moderate": BAND_MODERATE, "high": BAND_HIGH},
        "metrics": {k: metrics[k] for k in ("roc_auc", "f1", "precision", "recall", "accuracy",
                                            "pr_auc", "tp", "fp", "tn", "fn")},
        "n_train": n_train,
        "test_presentation": TEST_PRESENTATION,
        "features": feature_schema(ff, Xtr),
    }
    schema_path = art / "schema" / f"day{cutoff}.json"
    write_json(schema_path, schema)
    samples_path = art / "samples" / f"day{cutoff}.json"
    write_json(samples_path, sample_students(ff, s.samples_per_cutoff))
    return [model_path, schema_path, samples_path]


# --------------------------------------------------------------------------- main study
def run(data_dir: Path = DEFAULT_DATA_DIR, out_root: Path = ROOT, quick: bool = False) -> dict:
    s = Settings.quick_mode() if quick else Settings()
    started = time.time()
    art = out_root / "artifacts"
    gen = out_root / "web" / "src" / "generated"
    try:
        import pyarrow  # noqa: F401, PLC0415

        cache = data_dir.parent / "cache"
    except ImportError:
        cache = None
    log(f"loading OULAD from {data_dir} (quick={quick})")
    tables = load_tables(data_dir, cache_dir=cache)
    dataset, eda_out = eda(tables)
    log(f"dataset: {dataset}")

    features = {c: build_features(tables, c) for c in s.cutoffs}
    primary_ff = features[PRIMARY_CUTOFF]
    study = primary_study(primary_ff, s)
    Xtr, ytr, gtr, Xte, yte, _meta_te, _splits = study["data"]
    by_name = {r["name"]: r for r in study["records"]}

    # ---- timeline + deployed model per cutoff
    served: list[Path] = []
    timeline = []
    deployed_test_p = None
    for cutoff in s.cutoffs:
        ff = features[cutoff]
        a_Xtr, a_ytr, a_gtr, a_Xte, a_yte, _ = split(ff)
        a_splits = cv_splits(a_Xtr, a_ytr, a_gtr, s.n_folds)
        entry = {"cutoff": cutoff, "n_train": len(a_Xtr), "n_test": len(a_Xte),
                 "at_risk_rate_test": round(float(a_yte.mean()), 4),
                 "at_risk_remaining_test": int(a_yte.sum()), "models": {}}
        for name, est in timeline_estimators(study["best_params"], s).items():
            pipe = pipeline_for(name, est, ff.numeric, ff.categorical)
            rec, model, p, _ = evaluate(name, pipe, a_Xtr, a_ytr, a_Xte, a_yte, a_splits, s,
                                        ci=name == DEPLOY_MODEL)
            entry["models"][name] = {**{k: rec["test"][k] for k in (
                "roc_auc", "f1", "precision", "recall", "accuracy", "pr_auc", "tp", "fp", "fn",
                "tn")}, "threshold": rec["threshold"], "ci95": rec["ci95"]}
            if name == DEPLOY_MODEL:
                served += save_deployed(art, cutoff, model, ff, a_Xtr, rec["threshold"],
                                        rec["test"], len(a_Xtr), s)
                if cutoff == PRIMARY_CUTOFF:
                    deployed_model, deployed_test_p = model, p
                    deployed_threshold = rec["threshold"]
        timeline.append(entry)
        log(f"timeline day {cutoff}: " + ", ".join(
            f"{k}={v['roc_auc']:.3f}" for k, v in entry["models"].items()))

    # ---- ablation (deployed model family, feature groups)
    variants = {
        "Demographics only": FEATURE_GROUPS["demographic"],
        "+ Course & registration": FEATURE_GROUPS["demographic"] + FEATURE_GROUPS["course"]
        + FEATURE_GROUPS["registration"],
        "+ Assessments": FEATURE_GROUPS["demographic"] + FEATURE_GROUPS["course"]
        + FEATURE_GROUPS["registration"] + FEATURE_GROUPS["assessment"],
        "All features (+ VLE engagement)": list(primary_ff.X.columns),
        "All except protected attributes": [c for c in primary_ff.X.columns
                                            if c not in PROTECTED],
    }
    if s.quick:
        variants = {k: variants[k] for k in ("Demographics only",
                                             "All features (+ VLE engagement)")}
    abl_splits = cv_splits(Xtr, ytr, gtr, min(3, s.n_folds))
    ablation = []
    no_protected_p = None
    for label, cols in variants.items():
        num = [c for c in primary_ff.numeric if c in cols]
        cat = [c for c in primary_ff.categorical if c in cols]
        pipe = pipeline_for(DEPLOY_MODEL, deploy_estimator(study["best_params"]), num, cat)
        rec, _, p, _ = evaluate(label, pipe, Xtr[cols], ytr, Xte[cols], yte, abl_splits, s,
                                ci=False)
        ablation.append({"variant": label, "n_features": len(cols),
                         **{k: rec["test"][k] for k in ("roc_auc", "f1", "precision", "recall",
                                                        "accuracy")}})
        if label == "All except protected attributes":
            no_protected_p, no_protected_thr = p, rec["threshold"]
        log(f"ablation {label}: auc={rec['test']['roc_auc']:.4f}")

    # ---- importance
    log("permutation importance (deployed model, test set)")
    perm = permutation_importance(deployed_model, Xte, yte, scoring="roc_auc",
                                  n_repeats=s.perm_repeats, random_state=SEED)
    perm_rows = sorted(
        [{"feature": c, "label": FEATURE_LABELS.get(c, c), "mean": round(float(m), 5),
          "std": round(float(sd), 5)}
         for c, m, sd in zip(Xte.columns, perm.importances_mean, perm.importances_std,
                             strict=True)],
        key=lambda r: r["mean"], reverse=True)
    shap = None
    if "LightGBM (tuned)" in study["models"]:
        from riskml.research.explain import lgbm_shap  # noqa: PLC0415

        shap = lgbm_shap(study["models"]["LightGBM (tuned)"], Xte, primary_ff.categorical)
        for row in shap["mean_abs"]:
            row["label"] = FEATURE_LABELS.get(row["feature"], row["feature"])

    # ---- fairness
    fairness = {"threshold": deployed_threshold, "attributes": []}
    for attr in FAIRNESS_ATTRS:
        rows = group_fairness(yte, deployed_test_p, deployed_threshold, Xte[attr])
        entry = {"attribute": attr, "label": FEATURE_LABELS[attr], "groups": rows,
                 "gaps": fairness_gaps(rows)}
        if no_protected_p is not None:
            rows_np = group_fairness(yte, no_protected_p, no_protected_thr, Xte[attr])
            entry["gaps_without_protected"] = fairness_gaps(rows_np)
        fairness["attributes"].append(entry)

    # ---- manifest
    manifest = {
        "version": __version__,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__, "skops": skops.__version__,
        "data_sha256": OULAD_SHA256, "cutoffs": list(s.cutoffs),
        "primary_cutoff": PRIMARY_CUTOFF, "quick": s.quick,
        "files": {str(p.relative_to(art)): sha256_file(p) for p in sorted(served)},
    }
    write_json(art / "manifest.json", manifest)

    deployed_rec = by_name[study["deployed"]]
    champ_rec = by_name[study["champion"]]
    primary_timeline = next(e for e in timeline if e["cutoff"] == PRIMARY_CUTOFF)
    overview = {
        "generated_at": manifest["created_at"], "version": __version__, "quick": s.quick,
        "runtime_minutes": round((time.time() - started) / 60, 1),
        "dataset": dataset,
        "split": {"train_presentations": sorted(set(primary_ff.meta["code_presentation"])
                                                - {TEST_PRESENTATION}),
                  "test_presentation": TEST_PRESENTATION, "n_train": len(Xtr),
                  "n_test": len(Xte), "cv_folds": s.n_folds,
                  "cv": "StratifiedGroupKFold grouped by student"},
        "primary_cutoff": PRIMARY_CUTOFF,
        "cutoffs": list(s.cutoffs),
        "n_models": len(study["records"]),
        "n_features": primary_ff.X.shape[1],
        "champion": {"name": champ_rec["name"], "test": champ_rec["test"],
                     "cv_roc_auc": champ_rec["cv"]["roc_auc"], "ci95": champ_rec["ci95"]},
        "deployed": {"name": deployed_rec["name"], "test": deployed_rec["test"],
                     "cv_roc_auc": deployed_rec["cv"]["roc_auc"], "ci95": deployed_rec["ci95"],
                     "reason": "Best scikit-learn-native model: runs on the serverless "
                               "runtime without native OpenMP libraries and loads securely "
                               "with skops."},
        "early_warning": {"cutoff": PRIMARY_CUTOFF,
                          **primary_timeline["models"][DEPLOY_MODEL],
                          "at_risk_remaining_test": primary_timeline["at_risk_remaining_test"]},
        "notes": s.notes,
    }
    write_json(gen / "overview.json", overview)
    write_json(gen / "eda.json", eda_out)
    write_json(gen / "leaderboard.json", {
        "cutoff": PRIMARY_CUTOFF, "champion": study["champion"], "deployed": study["deployed"],
        "selection_metric": "cv.roc_auc", "models": study["records"],
        "tuning": study["tuning"]})
    write_json(gen / "curves.json", study["curves"])
    write_json(gen / "timeline.json", {"models": list(s.timeline_models),
                                       "deployed": DEPLOY_MODEL, "cutoffs": timeline})
    write_json(gen / "ablation.json", {"model": DEPLOY_MODEL, "variants": ablation})
    write_json(gen / "importance.json", {"permutation": perm_rows, "shap": shap,
                                         "shap_model": "LightGBM (tuned)" if shap else None})
    write_json(gen / "fairness.json", fairness)

    if not s.quick:
        try:
            from riskml.research.figures import make_figures  # noqa: PLC0415
            from riskml.research.report import write_report  # noqa: PLC0415

            make_figures(gen, out_root / "reports" / "figures")
            write_report(gen, out_root / "reports" / "results.md")
        except Exception as exc:  # figures are regenerable from the JSON; never lose a run
            log(f"figure/report generation failed ({exc!r}); rerun `python -m "
                "riskml.research.figures`")
    log(f"done in {(time.time() - started) / 60:.1f} min")
    return overview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.out_root, args.quick)


if __name__ == "__main__":
    main()
