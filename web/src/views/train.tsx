"use client";

import { Download, FileSpreadsheet, FileUp, Lock, Printer, RotateCcw, Search, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ChartCard, SERIES } from "@/components/charts/common";
import { ConfusionMatrix, HBarChart, UnitCurveChart } from "@/components/charts/charts";
import { Callout, Card, Container, PageHeader, Pill, RiskBadge, Segmented, Spinner, StatTile, buttonClass, cx } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { downloadCsv, toCsv } from "@/lib/csv";
import { fixed, formatPct, int } from "@/lib/format";
import type { Band, ColumnProfile, CustomPrediction, CustomResult, DatasetProfile, Meta, TrainConfig } from "@/lib/types";

type Step = "upload" | "map" | "training" | "results";
type Role = "risk" | "safe" | "unknown";
const ROLE_LABEL: Record<Role, string> = { risk: "At risk", safe: "Not at risk", unknown: "Unknown → score" };
const UNUSABLE: Record<string, string> = {
  id: "identifier",
  pii: "personal data — never used",
  text: "free text",
  constant: "same value everywhere",
  empty: "empty",
};

function StepIndicator({ step }: { step: Step }) {
  const steps: { key: Step; label: string }[] = [
    { key: "upload", label: "Upload" },
    { key: "map", label: "Map columns" },
    { key: "training", label: "Train" },
    { key: "results", label: "Results" },
  ];
  const idx = steps.findIndex((s) => s.key === step);
  return (
    <ol className="no-print mb-8 flex flex-wrap gap-2 text-sm" aria-label="Progress">
      {steps.map((s, i) => (
        <li key={s.key} aria-current={i === idx ? "step" : undefined}
          className={cx("flex items-center gap-2 rounded-full border px-3 py-1", i === idx ? "border-accent/40 bg-accent-wash text-accent-ink" : i < idx ? "border-line text-ink" : "border-line text-muted")}>
          <span className="tabular text-xs">{i + 1}</span> {s.label}
        </li>
      ))}
    </ol>
  );
}

function initialRoles(profile: DatasetProfile, target: string): Record<string, Role> {
  const col = profile.columns.find((c) => c.name === target);
  const out: Record<string, Role> = {};
  for (const v of col?.values ?? []) {
    out[v.value] = profile.suggested_positive.includes(v.value) && target === profile.suggested_target ? "risk"
      : profile.suggested_unknown.includes(v.value) && target === profile.suggested_target ? "unknown" : "safe";
  }
  return out;
}

export function TrainView() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [step, setStep] = useState<Step>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [target, setTarget] = useState("");
  const [roles, setRoles] = useState<Record<string, Role>>({});
  const [threshold, setThreshold] = useState<number>(40);
  const [idColumn, setIdColumn] = useState<string>("");
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [models, setModels] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<CustomResult | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.meta().then((m) => {
      setMeta(m);
      setModels(new Set(m.custom_models));
    }, () => undefined);
  }, []);

  useEffect(() => {
    if (step !== "training") return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [step]);

  const targetCol = profile?.columns.find((c) => c.name === target);
  const numericTarget = !!targetCol && targetCol.role === "numeric" && (targetCol.n_unique > 10 || !targetCol.values);

  async function handleFile(f: File) {
    setError(null);
    const max = meta?.limits.max_upload_bytes ?? 4_000_000;
    if (f.size > max) return setError(`This file is ${(f.size / 1e6).toFixed(1)} MB; the limit is ${(max / 1e6).toFixed(0)} MB. Remove unused columns or save as CSV.`);
    if (!/\.(csv|xlsx|txt)$/i.test(f.name)) return setError("Please upload a .csv or .xlsx file.");
    setBusy(true);
    try {
      const p = await api.profile(f);
      setFile(f);
      setProfile(p);
      const t = p.suggested_target ?? p.columns[p.columns.length - 1]?.name ?? "";
      setTarget(t);
      setRoles(initialRoles(p, t));
      setIdColumn(p.suggested_id ?? "");
      setExcluded(new Set());
      setStep("map");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not read the file.");
    } finally {
      setBusy(false);
    }
  }

  async function useSample() {
    setError(null);
    setBusy(true);
    try {
      await handleFile(await api.sampleFile());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the sample.");
      setBusy(false);
    }
  }

  const config: TrainConfig | null = useMemo(() => {
    if (!profile || !target) return null;
    const usable = profile.columns.filter((c) => !UNUSABLE[c.role] && c.name !== target && c.name !== idColumn);
    const base = {
      target,
      id_column: idColumn || null,
      exclude: usable.filter((c) => excluded.has(c.name)).map((c) => c.name),
      models: meta && models.size < meta.custom_models.length ? [...models] : null,
    };
    if (numericTarget) {
      return { ...base, positive_values: [], negative_values: null, unknown_values: [], positive_below: threshold };
    }
    const by = (r: Role) => Object.entries(roles).filter(([, v]) => v === r).map(([k]) => k);
    return { ...base, positive_values: by("risk"), negative_values: by("safe"), unknown_values: by("unknown"), positive_below: null };
  }, [profile, target, idColumn, excluded, models, meta, numericTarget, threshold, roles]);

  const configProblem = !config ? "Choose the result column." :
    !numericTarget && config.positive_values.length === 0 ? "Mark at least one value as 'At risk'." :
    !numericTarget && (config.negative_values?.length ?? 0) === 0 ? "Mark at least one value as 'Not at risk'." :
    target === idColumn ? "The result column cannot also be the ID column." :
    models.size === 0 ? "Select at least one model." : null;

  async function train() {
    if (!file || !config) return;
    setError(null);
    setStep("training");
    try {
      const r = await api.train(file, config);
      setResult(r);
      setStep("results");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Training failed.");
      setStep("map");
    }
  }

  function reset() {
    setStep("upload");
    setFile(null);
    setProfile(null);
    setResult(null);
    setError(null);
  }

  return (
    <>
      <PageHeader eyebrow="Bring your own data" title="Train risk models on your college's records">
        Upload past students with known results (pass / fail / detained / dropped) and current students whose result is not out yet.
        Every model is trained and compared on your data in seconds; current students get a risk score with reasons.
      </PageHeader>
      <Container className="mt-8">
        <StepIndicator step={step} />
        {error && <div className="mb-6"><Callout tone="warn" title="Please check">{error}</Callout></div>}

        {step === "upload" && (
          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            <Card>
              <label
                htmlFor="file"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files[0];
                  if (f) void handleFile(f);
                }}
                className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-line px-6 py-14 text-center hover:bg-surface-2"
              >
                {busy ? <Spinner className="h-6 w-6 text-accent" /> : <FileUp className="h-8 w-8 text-accent-ink" aria-hidden />}
                <span className="mt-4 text-base font-medium text-ink">Drop a CSV or Excel (.xlsx) file here, or click to choose</span>
                <span className="mt-1 text-sm text-ink-2">
                  Up to {int((meta?.limits.max_upload_bytes ?? 4e6) / 1e6)} MB · {int(meta?.limits.max_rows ?? 50000)} rows · {int(meta?.limits.max_cols ?? 200)} columns
                </span>
                <input id="file" ref={inputRef} type="file" className="sr-only" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) void handleFile(f); e.target.value = ""; }} />
              </label>
              <div className="mt-5 flex flex-wrap gap-3">
                <button type="button" className={buttonClass.primary} onClick={useSample} disabled={busy}>
                  <Sparkles className="h-4 w-4" aria-hidden /> Try the sample dataset
                </button>
                <a href="/api/custom/template" className={buttonClass.secondary}>
                  <FileSpreadsheet className="h-4 w-4" aria-hidden /> Download a blank template
                </a>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-muted">
                Sample: UCI &ldquo;Predict Students&rsquo; Dropout and Academic Success&rdquo; — 4,424 real higher-education students
                (Realinho et al., 2022, CC BY 4.0). Students still &ldquo;Enrolled&rdquo; are scored.
              </p>
            </Card>
            <div className="space-y-4">
              <Callout title="Privacy">
                <span className="flex items-start gap-2">
                  <Lock className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  Your file is processed in memory and discarded when the response is sent: nothing is stored or logged. Columns that look
                  like names, e-mails or phone numbers are detected and never used as model inputs.
                </span>
              </Callout>
              <Card>
                <h2 className="font-semibold text-ink">What a good file looks like</h2>
                <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
                  <li>One row per student, first row = column names.</li>
                  <li>A result column (e.g. Pass / Fail / Detained). Leave it blank for students whose result is not out yet.</li>
                  <li>Anything known early: attendance %, internal / mid-sem marks, assignment scores, previous SGPA, backlogs, LMS logins.</li>
                  <li>Optional roll number or ID column, used only to label rows.</li>
                </ul>
              </Card>
            </div>
          </div>
        )}

        {step === "map" && profile && (
          <div className="space-y-6">
            <Card>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="font-semibold text-ink">{file?.name}</p>
                  <p className="text-sm text-ink-2">{int(profile.n_rows)} rows · {profile.n_columns} columns</p>
                </div>
                <button type="button" className={buttonClass.ghost} onClick={reset}><RotateCcw className="h-4 w-4" aria-hidden /> Choose another file</button>
              </div>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <h2 className="font-semibold text-ink">1 · Which column holds the result?</h2>
                <select aria-label="Result column" className="mt-3 w-full rounded-lg border border-line bg-surface px-2.5 py-2 text-sm text-ink" value={target}
                  onChange={(e) => { setTarget(e.target.value); setRoles(initialRoles(profile, e.target.value)); }}>
                  {profile.columns.map((c) => <option key={c.name} value={c.name}>{c.name}{c.name === profile.suggested_target ? " (suggested)" : ""}</option>)}
                </select>
                {numericTarget ? (
                  <div className="mt-5">
                    <label htmlFor="thr" className="text-sm text-ink">Mark a student <strong>at risk</strong> when {target} is below</label>
                    <input id="thr" type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}
                      className="tabular ml-2 w-24 rounded-lg border border-line bg-surface px-2 py-1 text-sm text-ink" />
                    <p className="mt-2 text-xs text-muted">
                      Range in file: {targetCol?.stats?.min ?? "—"} to {targetCol?.stats?.max ?? "—"}. Blank cells are treated as &ldquo;result
                      not out yet&rdquo; and get scored.
                    </p>
                  </div>
                ) : (
                  <div className="mt-5">
                    <p className="mb-3 text-sm text-ink-2">Tell the model what each value means. Blank cells are always scored.</p>
                    <ul className="space-y-2">
                      {(targetCol?.values ?? []).map((v) => (
                        <li key={v.value} className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-sm text-ink">{v.value} <span className="text-muted">· {int(v.count)}</span></span>
                          <Segmented<Role> label={`Meaning of ${v.value}`} options={["risk", "safe", "unknown"]} value={roles[v.value] ?? "safe"}
                            onChange={(r) => setRoles((prev) => ({ ...prev, [v.value]: r }))} format={(r) => ROLE_LABEL[r]} />
                        </li>
                      ))}
                    </ul>
                    {!targetCol?.values && <p className="text-sm text-warning-ink">This column has too many distinct values to map; pick another.</p>}
                  </div>
                )}
              </Card>

              <Card>
                <h2 className="font-semibold text-ink">2 · Student ID column (optional)</h2>
                <select aria-label="ID column" className="mt-3 w-full rounded-lg border border-line bg-surface px-2.5 py-2 text-sm text-ink" value={idColumn} onChange={(e) => setIdColumn(e.target.value)}>
                  <option value="">None — label rows by spreadsheet row number</option>
                  {profile.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
                <h2 className="mt-6 font-semibold text-ink">3 · Models to compare</h2>
                <div className="mt-3 flex flex-wrap gap-2">
                  {meta?.custom_models.map((m) => (
                    <label key={m} className={cx("cursor-pointer rounded-full border px-3 py-1 text-sm", models.has(m) ? "border-accent/40 bg-accent-wash text-accent-ink" : "border-line text-ink-2")}>
                      <input type="checkbox" className="sr-only" checked={models.has(m)}
                        onChange={() => setModels((prev) => { const n = new Set(prev); if (n.has(m)) n.delete(m); else n.add(m); return n; })} />
                      {m}
                    </label>
                  ))}
                </div>
              </Card>
            </div>

            <Card>
              <h2 className="font-semibold text-ink">4 · Input columns</h2>
              <p className="mt-1 text-sm text-ink-2">Detected automatically. Untick anything that would not be known early in the term, or that encodes the result itself.</p>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[640px] text-sm">
                  <thead className="text-ink-2">
                    <tr>
                      <th scope="col" className="py-2 text-left font-medium">Use</th>
                      <th scope="col" className="py-2 text-left font-medium">Column</th>
                      <th scope="col" className="py-2 text-left font-medium">Detected as</th>
                      <th scope="col" className="py-2 text-right font-medium">Missing</th>
                      <th scope="col" className="py-2 text-left font-medium pl-4">Examples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile.columns.map((c: ColumnProfile) => {
                      const special = c.name === target ? "result column" : c.name === idColumn ? "ID column" : UNUSABLE[c.role];
                      return (
                        <tr key={c.name} className="border-t border-line">
                          <td className="py-2">
                            <input type="checkbox" aria-label={`Use ${c.name}`} disabled={!!special} checked={!special && !excluded.has(c.name)}
                              onChange={() => setExcluded((prev) => { const n = new Set(prev); if (n.has(c.name)) n.delete(c.name); else n.add(c.name); return n; })} />
                          </td>
                          <td className="py-2 text-ink">{c.name}</td>
                          <td className="py-2">{special ? <Pill>{special}</Pill> : <span className="text-ink-2">{c.role}</span>}</td>
                          <td className="tabular py-2 text-right text-ink-2">{c.missing_pct}%</td>
                          <td className="max-w-xs truncate py-2 pl-4 text-muted">{c.role === "pii" ? "hidden" : c.samples.slice(0, 3).join(", ")}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            <div className="flex flex-wrap items-center gap-4">
              <button type="button" className={buttonClass.primary} disabled={!!configProblem} onClick={train}>
                <Sparkles className="h-4 w-4" aria-hidden /> Train &amp; compare {models.size} models
              </button>
              {configProblem && <p className="text-sm text-warning-ink">{configProblem}</p>}
            </div>
          </div>
        )}

        {step === "training" && (
          <Card className="mx-auto max-w-xl text-center">
            <Spinner className="h-8 w-8 text-accent" />
            <p className="mt-4 text-lg font-semibold text-ink">Training and cross-validating {models.size} models…</p>
            <p className="mt-1 text-sm text-ink-2 tabular">{elapsed}s elapsed · budget {meta?.limits.time_budget_s ?? 40}s</p>
            <p className="mt-4 text-sm leading-relaxed text-ink-2">
              Each model is trained up to 5 times on different splits of your labeled rows to measure how well it generalises, then the best one
              scores the students whose result is unknown.
            </p>
          </Card>
        )}

        {step === "results" && result && <Results result={result} fileName={file?.name ?? "upload"} onReset={reset} />}
      </Container>
    </>
  );
}

function Results({ result, fileName, onReset }: { result: CustomResult; fileName: string; onReset: () => void }) {
  const r = result;
  const champ = r.leaderboard.find((m) => m.name === r.champion)!;
  const [source, setSource] = useState<"all" | "scored" | "cross-validated">(r.summary.n_scored > 0 ? "scored" : "all");
  const [band, setBand] = useState<"all" | Band>("all");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(50);
  const rows = useMemo(() => r.predictions.filter((p) =>
    (source === "all" || p.source === source) && (band === "all" || p.band === band) &&
    (!query || p.label.toLowerCase().includes(query.toLowerCase()))), [r, source, band, query]);

  function exportCsv() {
    const data = r.predictions.map((p: CustomPrediction) => ({
      row: p.row, student: p.label, risk_probability: p.probability, risk_band: p.band, flagged: p.at_risk ? "yes" : "no",
      actual: p.actual === null ? "" : p.actual === 1 ? "at risk" : "not at risk", source: p.source,
      reason_1: p.reasons[0] ? `${p.reasons[0].feature} = ${p.reasons[0].value ?? "missing"}` : "",
      reason_2: p.reasons[1] ? `${p.reasons[1].feature} = ${p.reasons[1].value ?? "missing"}` : "",
      reason_3: p.reasons[2] ? `${p.reasons[2].feature} = ${p.reasons[2].value ?? "missing"}` : "",
    }));
    downloadCsv(`risk-report-${fileName.replace(/\.[^.]+$/, "")}.csv`,
      toCsv(data, ["row", "student", "risk_probability", "risk_band", "flagged", "actual", "source", "reason_1", "reason_2", "reason_3"]));
  }

  return (
    <div className="space-y-6">
      <div className="no-print flex flex-wrap gap-3">
        <button type="button" className={buttonClass.primary} onClick={exportCsv}><Download className="h-4 w-4" aria-hidden /> Download risk report (CSV)</button>
        <button type="button" className={buttonClass.secondary} onClick={() => window.print()}><Printer className="h-4 w-4" aria-hidden /> Print / save as PDF</button>
        <button type="button" className={buttonClass.secondary} onClick={onReset}><RotateCcw className="h-4 w-4" aria-hidden /> Train on another file</button>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Students with a known result" value={int(r.summary.n_labeled)} hint={`${formatPct(r.summary.positive_rate)} at risk`} />
        <StatTile label="Students scored" value={int(r.summary.n_scored)} hint="result not out yet" />
        <StatTile label="Best model" value={<span className="text-lg">{r.champion}</span>} hint={`${r.summary.cv_folds}-fold CV ROC-AUC ${fixed(champ.roc_auc)}`} />
        <StatTile label="Training time" value={`${r.seconds.toFixed(1)}s`} hint={`${r.leaderboard.length} models compared`} />
      </div>

      {r.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}

      <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
        <table className="w-full min-w-[760px] text-sm">
          <caption className="px-4 pt-4 text-left font-semibold text-ink">Model comparison ({r.summary.cv_folds}-fold cross-validation on your labeled rows)</caption>
          <thead className="text-ink-2">
            <tr>
              {["Model", "ROC-AUC", "F1", "Recall", "Precision", "Accuracy", "MCC", "Time"].map((h) => (
                <th key={h} scope="col" className={cx("px-4 py-2.5 font-medium", h === "Model" ? "text-left" : "text-right")}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="tabular">
            {r.leaderboard.map((m) => (
              <tr key={m.name} className={cx("border-t border-line", m.name === r.champion && "bg-accent-wash/50")}>
                <td className="px-4 py-2.5 text-ink">{m.name} {m.name === r.champion && <Pill tone="accent">best</Pill>}</td>
                <td className="px-4 py-2.5 text-right text-ink">{fixed(m.roc_auc)} <span className="text-xs text-muted">±{fixed(m.roc_auc_std)}</span></td>
                <td className="px-4 py-2.5 text-right text-ink">{fixed(m.f1)}</td>
                <td className="px-4 py-2.5 text-right text-ink">{formatPct(m.recall)}</td>
                <td className="px-4 py-2.5 text-right text-ink">{formatPct(m.precision)}</td>
                <td className="px-4 py-2.5 text-right text-ink">{formatPct(m.accuracy)}</td>
                <td className="px-4 py-2.5 text-right text-ink">{fixed(m.mcc)}</td>
                <td className="px-4 py-2.5 text-right text-ink-2">{m.seconds.toFixed(1)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartCard title={`ROC curve — ${r.champion}`} subtitle="Out-of-fold predictions on your labeled rows."
          table={{ columns: ["False-positive rate", "True-positive rate"], rows: r.curves.roc.map(([a, b]) => [fixed(a), fixed(b)]) }}>
          <UnitCurveChart curves={[{ name: r.champion, color: SERIES[0]!, points: r.curves.roc }]} xLabel="False-positive rate" yLabel="True-positive rate" height={260} />
        </ChartCard>
        <ChartCard title="Confusion matrix" subtitle={`Threshold ${formatPct(r.threshold)} risk, out-of-fold.`}>
          <ConfusionMatrix {...r.confusion} />
        </ChartCard>
        <ChartCard title="What drives risk in your data" subtitle="Average change in risk when a column is set to its typical value."
          table={{ columns: ["Column", "Mean |impact|"], rows: r.importance.map((i) => [i.feature, fixed(i.mean_abs_impact, 4)]) }}>
          <HBarChart data={r.importance.slice(0, 10).map((i) => ({ label: i.feature, value: i.mean_abs_impact }))} format={(v) => formatPct(v, 1)} valueLabel="Mean |impact|" rowHeight={26} />
        </ChartCard>
      </div>

      <Card>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="font-semibold text-ink">Students, highest risk first</h2>
            <p className="mt-1 text-sm text-ink-2">
              &ldquo;Scored&rdquo; rows had no result and were predicted by the final model; &ldquo;cross-validated&rdquo; rows show the honest
              prediction the model made without having seen that student.
            </p>
          </div>
          <div className="no-print flex flex-wrap items-center gap-2">
            <Segmented<"all" | "scored" | "cross-validated"> label="Rows" options={["scored", "cross-validated", "all"]} value={source} onChange={setSource}
              format={(v) => ({ all: "All", scored: `Scored (${r.summary.n_scored})`, "cross-validated": "Known result" })[v]} />
            <Segmented<"all" | Band> label="Risk band" options={["all", "high", "moderate", "low"]} value={band} onChange={setBand}
              format={(v) => (v === "all" ? "Any risk" : v[0]!.toUpperCase() + v.slice(1))} />
            <label className="relative">
              <span className="sr-only">Search students</span>
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted" aria-hidden />
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search ID"
                className="w-40 rounded-lg border border-line bg-surface py-2 pl-8 pr-2 text-sm text-ink" />
            </label>
          </div>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead className="text-ink-2">
              <tr>
                {["Row", "Student", "Risk", "Band", "Actual", "Top reasons"].map((h) => (
                  <th key={h} scope="col" className={cx("py-2 font-medium", h === "Risk" || h === "Row" ? "text-right pr-4" : "text-left")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, limit).map((p) => (
                <tr key={`${p.source}-${p.row}`} className="border-t border-line align-top">
                  <td className="tabular py-2 pr-4 text-right text-muted">{p.row}</td>
                  <td className="py-2 text-ink">{p.label}</td>
                  <td className="tabular py-2 pr-4 text-right font-medium text-ink">{formatPct(p.probability, 0)}</td>
                  <td className="py-2"><RiskBadge band={p.band} /></td>
                  <td className="py-2 text-ink-2">{p.actual === null ? "—" : p.actual === 1 ? "At risk" : "Not at risk"}</td>
                  <td className="py-2 text-ink-2">
                    {p.reasons.length ? p.reasons.map((x) => `${x.feature}: ${x.value ?? "missing"} (typical ${x.typical ?? "—"})`).join(" · ") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && <p className="py-6 text-center text-sm text-ink-2">No students match these filters.</p>}
        </div>
        {rows.length > limit && (
          <button type="button" className={cx(buttonClass.secondary, "no-print mt-4")} onClick={() => setLimit((l) => l + 100)}>
            Show more ({int(rows.length - limit)} remaining)
          </button>
        )}
      </Card>

      <Card>
        <h2 className="font-semibold text-ink">Columns</h2>
        <p className="mt-2 text-sm text-ink-2"><strong>Used ({r.summary.features_used.length}):</strong> {r.summary.features_used.join(", ")}</p>
        {r.summary.excluded.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm text-ink-2">
            {r.summary.excluded.map((e) => <li key={e.column}><strong className="text-ink">{e.column}</strong> — {e.reason}</li>)}
          </ul>
        )}
      </Card>
    </div>
  );
}
