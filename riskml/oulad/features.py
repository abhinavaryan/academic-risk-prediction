"""Point-in-time feature engineering for early risk prediction.

For a prediction day ``cutoff`` (days since module start) we build one row per registration
that is still active on that day, using **only** events dated on or before the cutoff:

* demographic   - from studentInfo (known at registration)
* course        - module, semester (B = February start, J = October start), length
* registration  - how many days before the start the student registered
* vle           - Virtual Learning Environment click behaviour up to the cutoff
* assessment    - submissions and scores of assessments submitted up to the cutoff

The label is ``final_result in {Fail, Withdrawn}`` (1 = at risk). Students who had already
unregistered on or before the cutoff are excluded: they are no longer an early-warning target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from riskml.config import AT_RISK_RESULTS, DEFAULT_DATA_DIR, PRIMARY_CUTOFF
from riskml.oulad.load import COURSE_KEYS, KEYS, OuladTables, load_tables

ACTIVITY_TYPES = (
    "oucontent",
    "forumng",
    "quiz",
    "resource",
    "subpage",
    "homepage",
    "url",
    "ouwiki",
    "oucollaborate",
    "externalquiz",
    "page",
    "questionnaire",
)

DEMOGRAPHIC_CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
DEMOGRAPHIC_NUM = ["num_of_prev_attempts", "studied_credits"]
COURSE_CAT = ["code_module", "semester"]
COURSE_NUM = ["module_presentation_length"]
REGISTRATION_NUM = ["date_registration"]
VLE_NUM = [
    "vle_clicks_total",
    "vle_active_days",
    "vle_distinct_sites",
    "vle_clicks_precourse",
    "vle_clicks_last14",
    "vle_days_since_last",
    "vle_first_day",
    "vle_weekly_slope",
    "vle_clicks_rel_cohort",
    "vle_active_days_rel_cohort",
    *[f"vle_clicks_{a}" for a in ACTIVITY_TYPES],
    "vle_clicks_other",
]
ASSESS_NUM = [
    "assess_due",
    "assess_submitted",
    "assess_missed",
    "assess_submit_rate",
    "assess_mean_score",
    "assess_weighted_score",
    "assess_min_score",
    "assess_n_fail",
    "assess_mean_days_early",
    "assess_n_late",
    "assess_n_banked",
    "assess_tma_mean",
    "assess_cma_mean",
    "assess_score_rel_cohort",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "demographic": DEMOGRAPHIC_CAT + DEMOGRAPHIC_NUM,
    "course": COURSE_CAT + COURSE_NUM,
    "registration": REGISTRATION_NUM,
    "vle": VLE_NUM,
    "assessment": ASSESS_NUM,
}
CATEGORICAL = DEMOGRAPHIC_CAT + COURSE_CAT
NUMERIC = DEMOGRAPHIC_NUM + COURSE_NUM + REGISTRATION_NUM + VLE_NUM + ASSESS_NUM
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

FEATURE_LABELS: dict[str, str] = {
    "gender": "Gender",
    "region": "Region",
    "highest_education": "Highest education",
    "imd_band": "Deprivation (IMD) band",
    "age_band": "Age band",
    "disability": "Disability",
    "num_of_prev_attempts": "Previous attempts",
    "studied_credits": "Credits being studied",
    "code_module": "Module",
    "semester": "Start semester (B=Feb, J=Oct)",
    "module_presentation_length": "Course length (days)",
    "date_registration": "Registered (days relative to start)",
    "vle_clicks_total": "Total LMS clicks",
    "vle_active_days": "Active days on LMS",
    "vle_distinct_sites": "Distinct resources opened",
    "vle_clicks_precourse": "Clicks before course start",
    "vle_clicks_last14": "Clicks in last 14 days",
    "vle_days_since_last": "Days since last LMS activity",
    "vle_first_day": "First LMS activity (day)",
    "vle_weekly_slope": "Weekly click trend",
    "vle_clicks_rel_cohort": "Clicks vs. class median",
    "vle_active_days_rel_cohort": "Active days vs. class median",
    **{f"vle_clicks_{a}": f"Clicks: {a}" for a in ACTIVITY_TYPES},
    "vle_clicks_other": "Clicks: other activities",
    "assess_due": "Assessments due so far",
    "assess_submitted": "Assessments submitted",
    "assess_missed": "Assessments missed",
    "assess_submit_rate": "Submission rate",
    "assess_mean_score": "Average score",
    "assess_weighted_score": "Weighted average score",
    "assess_min_score": "Lowest score",
    "assess_n_fail": "Scores below 40",
    "assess_mean_days_early": "Avg. days submitted before deadline",
    "assess_n_late": "Late submissions",
    "assess_n_banked": "Banked (carried-over) results",
    "assess_tma_mean": "Tutor-marked avg. score",
    "assess_cma_mean": "Computer-marked avg. score",
    "assess_score_rel_cohort": "Avg. score vs. class median",
}


@dataclass(frozen=True)
class FeatureFrame:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    meta: pd.DataFrame
    categorical: list[str]
    numeric: list[str]
    cutoff: int


def _population(t: OuladTables, cutoff: int) -> pd.DataFrame:
    reg = t.registration.merge(t.info, on=KEYS, how="inner")
    reg = reg.merge(t.courses, on=COURSE_KEYS, how="left")
    active = reg["date_unregistration"].isna() | (reg["date_unregistration"] > cutoff)
    pop = reg.loc[active].sort_values(KEYS).reset_index(drop=True)
    pop["semester"] = pop["code_presentation"].str[-1]
    return pop


def _vle_features(t: OuladTables, pop: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    sv = t.student_vle
    sv = sv.loc[sv["date"] <= cutoff]
    sv = sv.merge(pop[KEYS], on=KEYS, how="inner")
    g = sv.groupby(KEYS, sort=False)
    out = pd.DataFrame(
        {
            "vle_clicks_total": g["sum_click"].sum(),
            "vle_active_days": g["date"].nunique(),
            "vle_distinct_sites": g["id_site"].nunique(),
            "vle_first_day": g["date"].min(),
            "vle_last_day": g["date"].max(),
        }
    )
    pre = sv.loc[sv["date"] < 0].groupby(KEYS, sort=False)["sum_click"].sum()
    last14 = sv.loc[sv["date"] > cutoff - 14].groupby(KEYS, sort=False)["sum_click"].sum()
    out["vle_clicks_precourse"] = pre
    out["vle_clicks_last14"] = last14

    act = sv["activity_type"].where(sv["activity_type"].isin(ACTIVITY_TYPES), "other")
    by_type = sv.assign(activity=act).pivot_table(
        index=KEYS, columns="activity", values="sum_click", aggfunc="sum", fill_value=0
    )
    for a in (*ACTIVITY_TYPES, "other"):
        out[f"vle_clicks_{a}"] = by_type[a] if a in by_type.columns else 0

    # Weekly trend: least-squares slope of weekly clicks over weeks 0..W-1 of the course so far.
    n_weeks = cutoff // 7 + 1
    if n_weeks >= 2:
        in_course = sv.loc[sv["date"] >= 0]
        week = in_course["date"] // 7
        sum_c = in_course.groupby(KEYS, sort=False)["sum_click"].sum()
        sum_wc = (
            (in_course["sum_click"] * week).groupby([in_course[k] for k in KEYS], sort=False).sum()
        )
        w = np.arange(n_weeks)
        sw, sww = w.sum(), (w * w).sum()
        slope = (n_weeks * sum_wc - sw * sum_c) / (n_weeks * sww - sw * sw)
        out["vle_weekly_slope"] = slope
    else:
        out["vle_weekly_slope"] = np.nan

    out = pop[KEYS].merge(out.reset_index(), on=KEYS, how="left")
    zero_fill = [
        "vle_clicks_total",
        "vle_active_days",
        "vle_distinct_sites",
        "vle_clicks_precourse",
        "vle_clicks_last14",
        *[f"vle_clicks_{a}" for a in (*ACTIVITY_TYPES, "other")],
    ]
    out[zero_fill] = out[zero_fill].fillna(0)
    if n_weeks >= 2:
        out["vle_weekly_slope"] = out["vle_weekly_slope"].fillna(0.0)
    out["vle_days_since_last"] = cutoff - out.pop("vle_last_day")

    cohort = out.groupby(COURSE_KEYS)
    med_clicks = cohort["vle_clicks_total"].transform("median")
    med_days = cohort["vle_active_days"].transform("median")
    out["vle_clicks_rel_cohort"] = (out["vle_clicks_total"] + 1) / (med_clicks + 1)
    out["vle_active_days_rel_cohort"] = (out["vle_active_days"] + 1) / (med_days + 1)
    return out


def _assessment_features(t: OuladTables, pop: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    a = t.assessments.loc[t.assessments["assessment_type"] != "Exam"]
    due = a.loc[a["date"] <= cutoff]
    n_due = due.groupby(COURSE_KEYS).size().rename("assess_due")

    sa = t.student_assessment.merge(a, on="id_assessment", how="inner")
    sa = sa.loc[sa["date_submitted"] <= cutoff].merge(pop[KEYS], on=KEYS, how="inner")
    sa = sa.assign(
        days_early=sa["date"] - sa["date_submitted"],
        late=(sa["date_submitted"] > sa["date"]).astype(int),
        failed=(sa["score"] < 40).astype(int),
        w_score=sa["score"] * sa["weight"],
        w_valid=sa["weight"].where(sa["score"].notna(), 0.0),
        on_due=(sa["date"] <= cutoff).astype(int),
    )
    g = sa.groupby(KEYS, sort=False)
    out = pd.DataFrame(
        {
            "assess_submitted": g.size(),
            "submitted_due": g["on_due"].sum(),
            "assess_mean_score": g["score"].mean(),
            "assess_min_score": g["score"].min(),
            "assess_n_fail": g["failed"].sum(),
            "assess_mean_days_early": g["days_early"].mean(),
            "assess_n_late": g["late"].sum(),
            "assess_n_banked": g["is_banked"].sum(),
            "w_sum": g["w_score"].sum(min_count=1),
            "w_total": g["w_valid"].sum(),
        }
    )
    for kind in ("TMA", "CMA"):
        sub = sa.loc[sa["assessment_type"] == kind].groupby(KEYS, sort=False)["score"].mean()
        out[f"assess_{kind.lower()}_mean"] = sub
    out = pop[KEYS].merge(out.reset_index(), on=KEYS, how="left")
    out = out.merge(n_due.reset_index(), on=COURSE_KEYS, how="left")
    for col in (
        "assess_due",
        "assess_submitted",
        "submitted_due",
        "assess_n_fail",
        "assess_n_late",
        "assess_n_banked",
    ):
        out[col] = out[col].fillna(0)
    out["assess_missed"] = (out["assess_due"] - out.pop("submitted_due")).clip(lower=0)
    out["assess_submit_rate"] = np.where(
        out["assess_due"] > 0,
        (out["assess_due"] - out["assess_missed"])
        / out["assess_due"].where(out["assess_due"] > 0, 1),
        np.nan,
    )
    w_total = out.pop("w_total")
    w_sum = out.pop("w_sum")
    out["assess_weighted_score"] = np.where(
        w_total > 0, w_sum / w_total.where(w_total > 0, 1), out["assess_mean_score"]
    )
    med = out.groupby(COURSE_KEYS)["assess_mean_score"].transform("median")
    out["assess_score_rel_cohort"] = out["assess_mean_score"] - med
    return out


def build_features(t: OuladTables, cutoff: int) -> FeatureFrame:
    pop = _population(t, cutoff)
    vle = _vle_features(t, pop, cutoff)
    assess = _assessment_features(t, pop, cutoff)
    df = pop.merge(vle, on=KEYS, how="left").merge(assess, on=KEYS, how="left")
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("object").where(X[col].notna(), np.nan)
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    y = df["final_result"].isin(AT_RISK_RESULTS).astype(int).rename("at_risk")
    meta = df[[*KEYS, "final_result"]].copy()
    return FeatureFrame(
        X=X,
        y=y,
        groups=df["id_student"].rename("group"),
        meta=meta,
        categorical=list(CATEGORICAL),
        numeric=list(NUMERIC),
        cutoff=cutoff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features and print a summary.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cutoff", type=int, default=PRIMARY_CUTOFF)
    args = parser.parse_args()
    tables = load_tables(args.data_dir, cache_dir=args.data_dir.parent / "cache")
    ff = build_features(tables, args.cutoff)
    print(
        f"cutoff={args.cutoff} rows={len(ff.X)} features={ff.X.shape[1]} "
        f"at_risk_rate={ff.y.mean():.3f}"
    )


if __name__ == "__main__":
    main()
