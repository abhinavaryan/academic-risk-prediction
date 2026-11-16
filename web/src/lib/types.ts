/** Shapes of the JSON written by the research pipeline (riskml/research/pipeline.py). */

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  mcc: number;
  balanced_accuracy: number;
  brier: number;
  log_loss: number;
  threshold: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}
export type CI = Partial<Record<keyof Metrics, [number, number]>>;

export interface ModelRecord {
  name: string;
  family: string;
  tuned: boolean;
  threshold: number;
  bases?: string[];
  cv: Pick<Metrics, "accuracy" | "precision" | "recall" | "f1" | "roc_auc" | "pr_auc" | "mcc" | "balanced_accuracy" | "brier"> & {
    roc_auc_std: number;
    f1_std: number;
    folds: { roc_auc: number; f1: number }[];
  };
  test: Metrics;
  test_at_0_5: Metrics;
  ci95: CI | null;
  timing: { cv_seconds: number | null; fit_seconds: number | null; predict_ms_per_1k: number | null };
  mcnemar_vs_champion: { statistic: number; p_value: number; a_only_correct: number; b_only_correct: number } | null;
  rank: number;
}

export interface Leaderboard {
  cutoff: number;
  champion: string;
  deployed: string;
  selection_metric: string;
  models: ModelRecord[];
  tuning: Record<string, { best_params: Record<string, number | string>; history: { trial: number; auc: number; best: number }[]; n_trials: number }>;
}

export type Curves = Record<string, { roc: [number, number][]; pr: [number, number][]; calibration: [number, number, number][] }>;

export type TimelineMetrics = Pick<Metrics, "roc_auc" | "f1" | "precision" | "recall" | "accuracy" | "pr_auc" | "tp" | "fp" | "fn" | "tn"> & {
  threshold: number;
  ci95: CI | null;
};

export interface Timeline {
  models: string[];
  deployed: string;
  cutoffs: {
    cutoff: number;
    n_train: number;
    n_test: number;
    at_risk_rate_test: number;
    at_risk_remaining_test: number;
    models: Record<string, TimelineMetrics>;
  }[];
}

export interface Ablation {
  model: string;
  variants: { variant: string; n_features: number; roc_auc: number; f1: number; precision: number; recall: number; accuracy: number }[];
}

export interface Importance {
  permutation: { feature: string; label: string; mean: number; std: number }[];
  shap: { mean_abs: { feature: string; value: number; label: string }[]; beeswarm: { feature: string; points: [number, number | null][] }[] } | null;
  shap_model: string | null;
}

export interface FairnessGroup {
  group: string;
  n: number;
  base_rate: number;
  positive_rate: number;
  tpr: number | null;
  fpr: number | null;
  precision: number | null;
  roc_auc: number | null;
}
export interface Gaps {
  tpr_gap: number | null;
  fpr_gap: number | null;
  positive_rate_gap: number | null;
}
export interface Fairness {
  threshold: number;
  attributes: { attribute: string; label: string; groups: FairnessGroup[]; gaps: Gaps; gaps_without_protected?: Gaps }[];
}

export interface RateRow {
  group: string;
  n: number;
  at_risk_rate: number;
}
export interface Eda {
  outcomes: { outcome: string; n: number }[];
  at_risk_rate: number;
  by_module: RateRow[];
  by_imd_band: RateRow[];
  by_highest_education: RateRow[];
  by_age_band: RateRow[];
  by_gender: RateRow[];
  by_disability: RateRow[];
  engagement_by_week: ({ week: number } & Record<string, number>)[];
  withdrawal_curve: { day: number; share: number }[];
  score_by_outcome: { outcome: string; n: number; p10: number; p25: number; median: number; p75: number; p90: number }[];
  presentations: { code_module: string; code_presentation: string; length_days: number; registrations: number }[];
}

export interface Overview {
  generated_at: string;
  version: string;
  quick: boolean;
  runtime_minutes: number;
  dataset: {
    registrations: number;
    students: number;
    modules: number;
    presentations: number;
    module_presentations: number;
    vle_interactions: number;
    vle_clicks: number;
    vle_resources: number;
    assessments: number;
    assessment_submissions: number;
  };
  split: { train_presentations: string[]; test_presentation: string; n_train: number; n_test: number; cv_folds: number; cv: string };
  primary_cutoff: number;
  cutoffs: number[];
  n_models: number;
  n_features: number;
  champion: { name: string; test: Metrics; cv_roc_auc: number; ci95: CI | null };
  deployed: { name: string; test: Metrics; cv_roc_auc: number; ci95: CI | null; reason: string };
  early_warning: TimelineMetrics & { cutoff: number; at_risk_remaining_test: number };
  notes: string[];
}

/* ---- API ---- */
export interface SchemaFeature {
  name: string;
  label: string;
  group: string;
  kind: "numeric" | "categorical";
  levels?: string[];
  mode?: string | null;
  integer?: boolean;
  median?: number | null;
  p01?: number;
  p99?: number;
  min?: number;
  max?: number;
  nullable: boolean;
}
export interface ModelSchema {
  cutoff: number;
  model: string;
  threshold: number;
  bands: { moderate: number; high: number };
  metrics: Partial<Metrics>;
  n_train: number;
  test_presentation: string;
  features: SchemaFeature[];
}
export type FeatureValue = number | string | null;
export interface Sample {
  id: string;
  final_result: string;
  at_risk: number;
  features: Record<string, FeatureValue>;
}
export interface Factor {
  name: string;
  label: string;
  value: FeatureValue;
  baseline: FeatureValue;
  impact: number;
  direction: "increases" | "decreases";
}
export type Band = "low" | "moderate" | "high";
export interface Prediction {
  cutoff: number;
  probability: number;
  band: Band;
  at_risk: boolean;
  threshold: number;
  factors: Factor[];
  filled_with_baseline: string[];
  model: string;
  model_version: string;
}

export interface ColumnProfile {
  name: string;
  role: "numeric" | "categorical" | "id" | "pii" | "text" | "constant" | "empty";
  missing_pct: number;
  n_unique: number;
  samples: string[];
  values?: { value: string; count: number }[];
  stats?: { min: number | null; median: number | null; max: number | null };
}
export interface DatasetProfile {
  n_rows: number;
  n_columns: number;
  columns: ColumnProfile[];
  suggested_target: string | null;
  suggested_positive: string[];
  suggested_negative: string[];
  suggested_unknown: string[];
  suggested_id: string | null;
}
export interface TrainConfig {
  target: string;
  positive_values: string[];
  negative_values: string[] | null;
  unknown_values: string[];
  positive_below: number | null;
  id_column: string | null;
  exclude: string[];
  models: string[] | null;
}
export interface CustomModelRow {
  name: string;
  family: string;
  roc_auc: number;
  roc_auc_std: number;
  f1: number;
  precision: number;
  recall: number;
  accuracy: number;
  pr_auc: number;
  mcc: number;
  balanced_accuracy: number;
  threshold: number;
  seconds: number;
}
export interface CustomPrediction {
  row: number;
  label: string;
  probability: number;
  band: Band;
  at_risk: boolean;
  actual: number | null;
  source: "scored" | "cross-validated";
  reasons: { feature: string; value: FeatureValue; typical: FeatureValue; impact: number; log_odds: number }[];
}
export interface CustomResult {
  summary: {
    n_rows: number;
    n_labeled: number;
    n_scored: number;
    n_ignored: number;
    n_positive: number;
    n_negative: number;
    positive_rate: number;
    cv_folds: number;
    features_used: string[];
    excluded: { column: string; reason: string }[];
    target: string;
  };
  leaderboard: CustomModelRow[];
  champion: string;
  threshold: number;
  confusion: { tp: number; fp: number; tn: number; fn: number };
  curves: { roc: [number, number][]; pr: [number, number][]; calibration: [number, number, number][] };
  importance: { feature: string; mean_abs_impact: number }[];
  predictions: CustomPrediction[];
  warnings: string[];
  seconds: number;
}
export interface Meta {
  version: string;
  cutoffs: number[];
  primary_cutoff: number;
  custom_models: string[];
  limits: { max_upload_bytes: number; max_rows: number; max_cols: number; time_budget_s: number };
}
