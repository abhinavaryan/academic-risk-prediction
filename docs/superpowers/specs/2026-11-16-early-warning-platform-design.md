# Agrim — Early Academic Risk Prediction Platform (Design)

Date: 2026-11-16 · Status: approved by owner ("make every change yourself")

## Intent

A 7th-semester B.Tech minor project (400 marks) that evaluators remember. It must feel
like a product built for real users, and every number it shows must be computed from
real data. Nothing may be hardcoded or synthetic.

## Goals

1. **Real research on OULAD.** Use all 7 tables (32,593 registrations, 28,785 students,
   10.6M VLE click rows). Predict *early* which students will Fail or Withdraw.
2. **Many models compared honestly.** About 15 algorithms, Optuna tuning,
   student-grouped CV, a train-on-past / test-on-future split, bootstrap confidence
   intervals, McNemar tests, calibration, explainability, fairness audit and a
   feature-group ablation.
3. **"How early can we predict?"** The same models are trained at prediction day
   0/30/60/90/120/150/180. Day 60 is the pre-registered primary cutoff.
4. **Live predictor.** Takes a real test-set student or a what-if edit, and returns a
   risk %, a risk band and the top factors behind the prediction.
5. **Bring-your-own-data training.** The evaluator uploads their college's CSV/XLSX.
   The app profiles the columns, then the user maps the target column and the
   at-risk values. Rows with a known result train and are cross-validated. Rows whose
   result is blank or "unknown" (e.g. "Enrolled") get scored. The output is a
   leaderboard, per-student risk with reasons, and a CSV/printable report. Uploaded
   data is processed in memory and never stored.
6. **Production-grade engineering.** Input validation, upload hardening, no untrusted
   deserialisation, security headers, rate limiting, bounded concurrency, tests,
   CI, docs, a model card.
7. **Free hosting.** The primary host is Vercel. Hugging Face Spaces (Docker) is a
   backup. It must also run on localhost and in GitHub Codespaces.

## Non-goals

User accounts or login, a database or persistence, GPU training, a mobile app.

## Anti-goals

- Any fabricated metric, chart value, testimonial or dataset row shown as real.
- A model that silently accepts typos. The old free-text form did this.
- Data leakage: future clicks or scores used for an early prediction, or the same
  student appearing in both train and validation folds.

## Constraints

- **Vercel Python functions:**
  - 500 MB unzipped bundle
  - 4.5 MB request body limit, so uploads are capped at 4 MB
  - Hobby plan `maxDuration` is 60 s (set to 60), so custom training runs on a time
    budget
- **LightGBM** needs the system `libgomp.so.1`, which the serverless runtime may lack.
  **CatBoost** is 268 MB installed. Both are therefore *research-only*. The serving
  stack is scikit-learn, which vendors its own OpenMP.
- **Models must be served as scikit-learn.** The deployed models are
  `HistGradientBoostingClassifier` (one per cutoff). They are loaded only from the
  repo, after SHA-256 verification against `artifacts/manifest.json`. Files uploaded
  by users are never deserialised.
- **OULAD raw data (464 MB) is not committed.** The pipeline downloads it from UCI and
  verifies its SHA-256 (`f2ed1902…cc83a65d`).
- **The UCI "Predict Students' Dropout and Academic Success" dataset** (4,424
  students, CC BY 4.0) is committed as the sample file for the custom-training demo.

## Approaches considered

| Approach | Trade-off | Verdict |
|---|---|---|
| A. Next.js static UI + FastAPI Python functions, one Vercel project. The same FastAPI app also serves the built UI in Docker/locally | One codebase, one URL; scikit-learn fits in 500 MB; cold start a few seconds | **Chosen** |
| B. Vercel UI + API on Hugging Face Spaces only | Persistent server, but two accounts and sleeping containers | Used only as backup deployment of A |
| C. Static site + Pyodide (Python in browser) | No server, private by default; but a ~40 MB first load, slow training, hard to defend in viva | Rejected |

## Problem formulation (OULAD)

- **Unit:** one registration (student × module-presentation).
- **Label:** `at_risk = final_result ∈ {Fail, Withdrawn}`. Positive class = at-risk.
- **Population at cutoff D:** registrations not yet unregistered (`date_unregistration`
  missing or > D).
- **Features at D:** only events with `date ≤ D`. Assessment scores count once
  `date_submitted ≤ D`; exams are excluded.
  - demographics (8)
  - course context: module, B/J semester, length
  - registration timing
  - VLE: totals, active days, distinct resources, per-activity-type clicks,
    last-14-days clicks, days since last activity, pre-course clicks, weekly trend,
    cohort-relative clicks
  - assessments: due, submitted, missed, mean/weighted/min score, fails, lateness,
    banked
- **Split:**
  - Test = presentation 2014J (the latest). Train = 2013B, 2013J, 2014B.
  - Model selection uses `StratifiedGroupKFold(5)` on train, grouped by `id_student`.
  - The decision threshold is tuned on out-of-fold predictions, never on test.
- **Selection metric:** CV ROC-AUC. F1, precision, recall, accuracy, PR-AUC, MCC,
  balanced accuracy and Brier score are also reported.

## Decided by owner

Vercel stays primary, with an alternative host as backup. Real OULAD data only. Many
models. Custom training on college data. Everything free. Claude makes all calls.

## Left to the builder

Visual design, exact feature list, hyper-parameter search spaces, chart choices and
copy, all within the anti-goals above.
