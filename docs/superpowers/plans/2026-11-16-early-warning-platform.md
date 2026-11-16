# Agrim Early-Warning Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crashing single-model demo with a research-grade, production-hardened early-warning platform. It covers real OULAD research, a live predictor, and bring-your-own-data training.

**Architecture:**
- A Python package `riskml` holds:
  - the OULAD loaders and point-in-time feature builder
  - the research pipeline: it writes `artifacts/` (served models + schema) and `web/src/generated/` (report JSON)
  - the custom-training engine
  - a FastAPI app
- `api/index.py` exposes that app to Vercel. A Next.js static-export UI in `web/` is served by Vercel's CDN, or by FastAPI itself in Docker/locally.

**Tech Stack:**
- Python 3.12/3.13: pandas, numpy, scikit-learn 1.9.1, FastAPI, pydantic v2, openpyxl, skops
- Research-only: LightGBM, XGBoost, CatBoost, Optuna, matplotlib
- Web: Next.js (App Router, static export), TypeScript, Tailwind v4, Recharts, Vitest, Playwright

**Spec:** `docs/superpowers/specs/2026-11-16-early-warning-platform-design.md`

## Global Constraints

- **Serving dependencies** (`requirements.txt`) must not include lightgbm, xgboost, catboost, optuna or matplotlib.
- **Exact pins:** `scikit-learn==1.9.1` in both `requirements.txt` and `requirements-research.txt`.
- **Cutoffs:** `CUTOFFS = (0, 30, 60, 90, 120, 150, 180)`. `PRIMARY_CUTOFF = 60`. `SEED = 42`.
- **Splits:** `TEST_PRESENTATION = "2014J"`. `at_risk` = `final_result in {"Fail","Withdrawn"}`.
- **Upload limits:**
  - `MAX_UPLOAD_BYTES = 4_000_000`, `MAX_ROWS = 50_000`, `MAX_COLS = 200`
  - `MIN_LABELED_ROWS = 30`, `MIN_CLASS_COUNT = 10`
  - xlsx uncompressed ≤ 60 MB
- **Training:** `CUSTOM_TIME_BUDGET_S = 40`. Concurrent custom trainings per instance ≤ 2; when the limit is hit, return 503 with `Retry-After`.
- **Deserialisation:** never unpickle or deserialise uploaded content. Served artifacts are loaded only after a SHA-256 match with `artifacts/manifest.json`.
- **No fabricated numbers:** every metric in the UI is imported from `web/src/generated/*.json`, which the pipeline writes.
- **Excel-safe exports:** CSV cells starting with `= + - @ \t \r` are prefixed with `'` on export.

## Review Focus

1. **A teacher's real file is messy:**
   - BOM, `;` separators, tabs in headers, Windows-1252 encoding
   - Excel with merged header rows
   - an ID column, a "Name" column
   
   Expected: it parses, IDs and PII are excluded from features automatically, and names are kept for display only. Test: `tests/custom/test_io.py::test_reads_uci_semicolon_bom_tab_header`, `test_cp1252`, `test_pii_and_id_columns_excluded`.
2. **The target column leaks the answer** (e.g. "Final Grade" next to "Result"). Expected: a near-perfect single-feature predictor is auto-excluded and reported as a warning. Test: `tests/custom/test_trainer.py::test_leaky_column_excluded_with_warning`.
3. **Two people press Train at once on one instance.** Expected: both succeed, or one gets a clean 503; results are deterministic and isolated. Test: `tests/server/test_concurrency.py::test_parallel_custom_training_isolated`.
4. **Predict request with a misspelt category or an out-of-range number.** Expected: 422 with a field-level message, never a silent default. Test: `tests/server/test_predict.py::test_unknown_category_rejected`, `test_negative_clicks_rejected`.
5. **Early-cutoff leakage:** a click or score dated after D changes the features at D. Expected: features are identical whether or not future rows exist. Test: `tests/oulad/test_features.py::test_no_future_information`.

---

### Task 1: Repo foundation

**Files:**
- Delete: `model/`, `templates/`
- Create: `riskml/__init__.py`, `riskml/config.py`
- Create: `requirements.txt`, `requirements-research.txt`, `requirements-dev.txt`, `ruff.toml`, `pytest.ini`, `Makefile`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `riskml.config`: `CUTOFFS`, `PRIMARY_CUTOFF`, `SEED`, `TEST_PRESENTATION`, `ROOT`, `ARTIFACTS_DIR`, `GENERATED_DIR`, and the limits above
  - `riskml.__version__ = "1.0.0"`

- [ ] Write the config and requirement files, delete the 95 MB pickle and the template.
- [ ] Verify with `ruff check . && pytest -q` (no tests collected yet = OK). Commit.

### Task 2: OULAD loading and point-in-time features

**Files:**
- Create: `riskml/oulad/{__init__,download,load,features}.py`
- Test: `tests/oulad/test_load.py`, `tests/oulad/test_features.py`

**Interfaces:**
- `download_oulad(dest: Path) -> Path`: UCI URL; SHA-256 checked; extracts the files.
- `load_tables(data_dir: Path, cache_dir: Path | None) -> OuladTables`: frozen dataclass with fields `info, registration, assessments, student_assessment, vle, student_vle, courses`. Cleaning rules:
  - `?` becomes NaN
  - `imd_band` `"10-20"` becomes `"10-20%"`
  - integer dtypes are downcast
  - `student_vle` is merged with `activity_type`
- `build_features(t: OuladTables, cutoff: int) -> FeatureFrame`: a dataclass with
  - `X: pd.DataFrame`
  - `y: pd.Series[int]`
  - `groups: pd.Series` (id_student)
  - `meta: pd.DataFrame` (code_module, code_presentation, id_student, final_result)
  - `categorical: list[str]`, `numeric: list[str]`
- `FEATURE_GROUPS: dict[str, list[str]]`, keyed by `demographic`, `course`, `registration`, `vle`, `assessment`.

**Tests** (small hand-built `OuladTables` fixture in `tests/oulad/conftest.py`):
- `test_no_future_information`: appending rows dated `cutoff+1` leaves `X` unchanged (`pd.testing.assert_frame_equal`).
- `test_population_excludes_already_withdrawn`: a student with `date_unregistration=10` is absent at cutoff 30 and present at cutoff 0.
- `test_missed_assessments_counted`: a deadline at day 20 with no submission gives `assess_missed == 1` at cutoff 30.
- `test_imd_band_normalised`.

- [ ] Write the tests and the fixture. Run them; they fail with ImportError.
- [ ] Implement. Run `pytest tests/oulad -q`; it passes. Also smoke-test on the real data: `python -m riskml.oulad.features --data-dir … --cutoff 60` prints a shape of about 26k × ~70. Commit.

### Task 3: Model zoo, metrics, tuning, explain, fairness

**Files:**
- Create: `riskml/modeling/{__init__,zoo,metrics,fairness}.py`, `riskml/research/{__init__,tuning,explain}.py`
- Test: `tests/modeling/test_metrics.py`, `tests/modeling/test_zoo.py`

**Interfaces:**
- `make_preprocessor(numeric, categorical, scale: bool) -> ColumnTransformer`: median imputation + missing indicator; one-hot with `handle_unknown="ignore"`, `min_frequency=0.01`.
- `research_zoo(numeric, categorical, seed) -> dict[str, Pipeline]`. Models:
  - Dummy, LogReg, NaiveBayes, KNN, DecisionTree, RandomForest, ExtraTrees, AdaBoost
  - HistGB, LinearSVM(calibrated), MLP
  - LightGBM, XGBoost, CatBoost (each imported lazily)
- `serving_zoo(numeric, categorical, seed) -> dict[str, Pipeline]`: sklearn-only subset.
- `binary_metrics(y, p, threshold) -> dict`. Keys: `accuracy, precision, recall, f1, roc_auc, pr_auc, mcc, balanced_accuracy, brier, log_loss, threshold, tp, fp, tn, fn`.
- `best_f1_threshold(y, p) -> float`
- `bootstrap_ci(y, p, threshold, n=1000, seed) -> dict[metric, (lo, hi)]`
- `mcnemar(y, pred_a, pred_b) -> {statistic, p_value}`
- `curve_points(y, p, n=101) -> {roc:[[fpr,tpr]], pr:[[recall,precision]], calibration:[[mean_pred, frac_pos, count]]}`
- `group_fairness(y, p, threshold, groups: pd.Series) -> list[dict]`

**Tests:**
- `test_perfect_predictions_metrics` (f1 = 1, auc = 1)
- `test_best_threshold_beats_half_on_skewed_scores`
- `test_bootstrap_ci_contains_point_estimate`
- `test_mcnemar_identical_preds_p_is_1`
- `test_serving_zoo_has_no_research_only_imports` (asserts `"lightgbm" not in sys.modules` after constructing it)

- [ ] Tests → fail → implement → pass → commit.

### Task 4: Research pipeline run and artifacts

**Files:**
- Create: `riskml/research/{pipeline,figures,report}.py`
- Output: `artifacts/models/hgb_day{D}.skops`, `artifacts/schema/day{D}.json`, `artifacts/samples/day{D}.json`, `artifacts/manifest.json`
- Output: `web/src/generated/{overview,leaderboard,timeline,ablation,importance,fairness,eda,curves}.json`
- Output: `reports/figures/*.png`, `reports/results.md`

**Interfaces:**
- `run(data_dir, out_root, quick: bool=False) -> None`. CLI: `python -m riskml.research.pipeline --data-dir DIR [--quick]`.
- Schema JSON for each feature: `{name, kind: "numeric"|"categorical", levels?, min, max, median, label, group}`.
- Samples JSON: 150 real 2014J registrations (stratified) with `{id, features, actual}`.

- [ ] Implement. `--quick` (≤ 2 min, used by CI) runs 3 models, 2 folds, no tuning. The full run uses Optuna with 30 trials for HistGB, LightGBM, XGBoost and RandomForest. It runs every model at day 60, LogReg/RF/HistGB/LightGBM at all cutoffs, and the ablation, bootstrap, McNemar, fairness and importance steps.
- [ ] Run the full pipeline on the real data and check that the outputs exist and the manifest hashes match. Commit the artifacts and the generated JSON.

### Task 5: Serving registry and predictor

**Files:**
- Create: `riskml/serving/{__init__,registry,predictor}.py`
- Test: `tests/serving/test_registry.py`, `tests/serving/test_predictor.py`

**Interfaces:**
- `ModelRegistry(artifacts_dir).get(cutoff) -> LoadedModel(model, schema, threshold, version)`. Lazy loading; thread-safe through a double-checked lock. Raises `ArtifactIntegrityError` on a hash mismatch.
- `predict(loaded, features: dict) -> Prediction(probability, band, at_risk, threshold, factors: list[Factor(name,label,value,impact)])`.
- Bands: `low < 0.33 ≤ moderate < 0.66 ≤ high`.
- Factors are baseline-replacement attributions computed against the schema medians/modes, top 6 by |impact|.

**Tests:**
- `test_tampered_artifact_refused`
- `test_concurrent_get_loads_once` (32 threads, the loader is called once)
- `test_prediction_deterministic`
- `test_factors_sorted_by_abs_impact`

- [ ] Tests → fail → implement → pass → commit.

### Task 6: Custom training engine

**Files:**
- Create: `riskml/custom/{__init__,io,profile,trainer}.py`
- Add: `sample_data/uci_student_dropout.csv` and its README attribution
- Add: `sample_data/college_template.csv` (headers only)
- Test: `tests/custom/test_io.py`, `test_profile.py`, `test_trainer.py`

**Interfaces:**
- `read_table(data: bytes, filename: str) -> pd.DataFrame`. Raises `UploadError(code, message)`. Handles CSV delimiter sniffing, encodings `utf-8-sig/utf-8/cp1252/latin-1`, and xlsx with a zip-bomb guard. Normalises headers.
- `profile(df) -> DatasetProfile`. Contains `columns[{name, role: numeric|categorical|id|pii|text|constant, missing_pct, n_unique, samples}]`, `suggested_target`, `suggested_positive: list[str]`, `suggested_unknown: list[str]`, `suggested_id`.
- `train_custom(df, cfg: TrainConfig, time_budget_s) -> CustomResult`. `TrainConfig` fields: `target, positive_values, negative_values, unknown_values, id_column, exclude`. `CustomResult` fields: `leaderboard, champion, threshold, curves, confusion, importance, predictions[{row_id, label, probability, band, at_risk, actual, reasons}], warnings, timings`.

**Tests:**
- `test_reads_uci_semicolon_bom_tab_header` (4424 × 37, first column `Marital status`)
- `test_cp1252`, `test_xlsx_roundtrip`, `test_zip_bomb_rejected`, `test_oversize_rejected`, `test_empty_rejected`
- `test_pii_and_id_columns_excluded`
- `test_profile_suggests_target_on_uci` (`Target`, positive `Dropout`, unknown `Enrolled`)
- `test_trains_on_uci_and_scores_enrolled` (794 scored rows, champion ROC-AUC > 0.85)
- `test_leaky_column_excluded_with_warning`
- `test_single_class_rejected`
- `test_deterministic`

- [ ] Tests → fail → implement → pass → commit.

### Task 7: FastAPI server with hardening

**Files:**
- Create: `riskml/server/{__init__,main,schemas,security,routes}.py`, `api/index.py`
- Test: `tests/server/test_meta.py`, `test_predict.py`, `test_custom.py`, `test_security.py`, `test_concurrency.py`

**Interfaces** (all under `/api`):
- `GET /api/health` → `{status, version, models_loaded}`
- `GET /api/schema/{cutoff}`
- `GET /api/samples/{cutoff}`
- `POST /api/predict {cutoff, features}` → `Prediction`
- `POST /api/custom/profile` (multipart `file`)
- `POST /api/custom/train` (multipart `file` + `config` JSON string)
- Docs at `/api/docs`
- Middleware: request-id, security headers, body-size cap, per-IP token bucket on `/api/custom/*` (20/min). Training calls run in the threadpool behind `BoundedSemaphore(2)`.
- With `SERVE_STATIC=1`, the app mounts `web/out` at `/`.

**Tests:**
- `test_unknown_category_rejected`, `test_negative_clicks_rejected`, `test_extra_field_rejected`
- `test_security_headers_present`, `test_errors_do_not_leak_tracebacks`
- `test_rate_limit_429`, `test_busy_returns_503`
- `test_parallel_custom_training_isolated`, `test_parallel_predict_consistent` (50 threads)
- `test_profile_and_train_uci_end_to_end`

- [ ] Tests → fail → implement → pass → commit.

### Task 8: Next.js web app

**Files:**
- Create: `web/` with Next.js app routes `/` (overview), `/dataset`, `/models`, `/predict`, `/train`, `/methodology`
- Create: `web/src/lib/{api,format,csv}.ts`, `web/src/components/*`
- Test: `web/src/lib/csv.test.ts`, `web/src/lib/format.test.ts`

**Interfaces:**
- Imports `@/generated/*.json`. Calls `/api/*` via `api.ts` with `NEXT_PUBLIC_API_BASE` (default `""`).
- `toCsv(rows, columns) -> string` neutralises formulas.

**Tests:**
- `toCsv escapes =cmd`, `quotes commas/newlines`
- `formatPct(0.1234) === "12.3%"`

- [ ] Build the pages. Run `npm run lint && npm run typecheck && npm test && npm run build`. Commit.

### Task 9: End-to-end tests, deployment, CI, docs

**Files:**
- Create: `web/e2e/*.spec.ts`, `playwright.config.ts`
- Create: `vercel.json`, `.vercelignore`, `Dockerfile`, `.dockerignore`, `.devcontainer/devcontainer.json`
- Create: `.github/workflows/{ci,retrain}.yml`, `.github/dependabot.yml`
- Create: `README.md`, `docs/{METHODOLOGY,MODEL_CARD,DEPLOYMENT,VIVA_GUIDE,DEMO_SCRIPT,SECURITY}.md`, `notebooks/colab_pipeline.ipynb`

**E2E tests:**
- `overview shows real dataset counts`
- `predict sample student shows risk`
- `train on sample data shows leaderboard and downloads csv`

- [ ] Run e2e against `SERVE_STATIC=1 uvicorn` locally. Build the Docker image if docker is available. Take real screenshots for the README. Commit and push.
