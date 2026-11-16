"use client";

import { Dice5, Eye, Lightbulb, RotateCcw, UserRound } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DivergingBars, RiskMeter } from "@/components/charts/charts";
import { Callout, Card, Container, PageHeader, Pill, RiskBadge, Segmented, Spinner, buttonClass, cx } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { CONTEXT_GROUPS, FEATURE_GROUP_LABELS } from "@/lib/data";
import { formatPct } from "@/lib/format";
import type { FeatureValue, ModelSchema, Prediction, Sample, SchemaFeature } from "@/lib/types";

const KEY_FEATURES = [
  "vle_clicks_total", "vle_active_days", "vle_clicks_last14", "vle_days_since_last",
  "assess_submitted", "assess_missed", "assess_mean_score", "assess_n_late",
  "num_of_prev_attempts", "studied_credits", "highest_education", "imd_band", "age_band", "code_module", "disability",
];

const ACTIONS: { test: (f: string) => boolean; text: string }[] = [
  { test: (f) => /^vle_(clicks|active|distinct|days_since|weekly)/.test(f), text: "Low LMS engagement: send a personal check-in and point to this week's key materials." },
  { test: (f) => /^assess_(missed|submit_rate|submitted)$/.test(f), text: "Missing work: agree a catch-up plan for outstanding assessments." },
  { test: (f) => /^assess_(mean_score|weighted_score|min_score|n_fail|tma_mean|cma_mean|score_rel)/.test(f), text: "Low scores: offer a tutoring or revision session on recent topics." },
  { test: (f) => /^assess_(n_late|mean_days_early)$/.test(f), text: "Late submissions: help with time planning and send deadline reminders." },
  { test: (f) => f === "num_of_prev_attempts", text: "Repeat attempt: pair the student with a mentor and review what went wrong last time." },
  { test: (f) => f === "studied_credits", text: "Heavy study load: discuss workload and whether to reduce credits this term." },
];

function fmtValue(f: SchemaFeature | undefined, v: FeatureValue): string {
  if (v === null || v === undefined) return "missing";
  if (typeof v === "string") return v;
  if (f?.name.includes("rate")) return formatPct(v, 0);
  return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(2);
}

function NumberField({ f, value, onChange }: { f: SchemaFeature; value: FeatureValue; onChange: (v: number | null) => void }) {
  const lo = f.p01 ?? f.min ?? 0;
  const hi = Math.max(f.p99 ?? f.max ?? 1, lo + 1);
  const step = f.integer ? 1 : Math.max((hi - lo) / 100, 0.01);
  const num = typeof value === "number" ? value : null;
  const id = `f-${f.name}`;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm text-ink">{f.label}</label>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          className="tabular w-24 rounded-lg border border-line bg-surface px-2 py-1 text-right text-sm text-ink"
          value={num ?? ""}
          placeholder="missing"
          min={f.min}
          max={f.max}
          step={step}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") return onChange(f.nullable ? null : (f.median ?? 0));
            const n = Number(raw);
            if (Number.isFinite(n)) onChange(Math.min(f.max ?? n, Math.max(f.min ?? n, n)));
          }}
        />
      </div>
      <input
        aria-label={`${f.label} slider`}
        type="range"
        className="mt-2 w-full accent-[var(--accent)]"
        min={lo}
        max={hi}
        step={step}
        value={num ?? f.median ?? lo}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function SelectField({ f, value, onChange }: { f: SchemaFeature; value: FeatureValue; onChange: (v: string | null) => void }) {
  const id = `f-${f.name}`;
  return (
    <div>
      <label htmlFor={id} className="text-sm text-ink">{f.label}</label>
      <select
        id={id}
        className="mt-1.5 w-full rounded-lg border border-line bg-surface px-2.5 py-2 text-sm text-ink"
        value={typeof value === "string" ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
      >
        {f.nullable && <option value="">Unknown</option>}
        {f.levels?.map((l) => (
          <option key={l} value={l}>{l}</option>
        ))}
      </select>
    </div>
  );
}

export function PredictView() {
  const [cutoffs, setCutoffs] = useState<number[]>([]);
  const [cutoff, setCutoff] = useState<number | null>(null);
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleId, setSampleId] = useState<string>("");
  const [features, setFeatures] = useState<Record<string, FeatureValue>>({});
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const seq = useRef(0);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    api.meta().then(
      (m) => {
        setCutoffs(m.cutoffs);
        setCutoff(m.primary_cutoff);
      },
      (e: ApiError) => setError(e.message),
    );
  }, []);

  const baselineOf = useCallback((s: ModelSchema) => {
    const out: Record<string, FeatureValue> = {};
    for (const f of s.features) out[f.name] = f.kind === "numeric" ? (f.median ?? null) : (f.mode ?? null);
    return out;
  }, []);

  // Load schema + real students whenever the prediction day changes; keep the same student if possible.
  useEffect(() => {
    if (cutoff === null) return;
    let cancelled = false;
    Promise.all([api.schema(cutoff), api.samples(cutoff)]).then(
      ([s, smp]) => {
        if (cancelled) return;
        setSchema(s);
        setSamples(smp);
        const keep = smp.find((x) => x.id === sampleId);
        const pick = keep ?? smp[0];
        setSampleId(pick ? pick.id : "");
        setFeatures(pick ? { ...baselineOf(s), ...pick.features } : baselineOf(s));
        setRevealed(false);
      },
      (e: ApiError) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cutoff, baselineOf]);

  // Debounced prediction; stale responses are dropped (sequence number + AbortController).
  useEffect(() => {
    if (!schema || cutoff === null || schema.cutoff !== cutoff) return;
    const id = ++seq.current;
    const t = setTimeout(() => {
      abort.current?.abort();
      const ctrl = new AbortController();
      abort.current = ctrl;
      setLoading(true);
      api
        .predict(cutoff, features, ctrl.signal)
        .then((p) => {
          if (id === seq.current) {
            setPrediction(p);
            setError(null);
          }
        })
        .catch((e: unknown) => {
          if (id !== seq.current || (e instanceof DOMException && e.name === "AbortError")) return;
          if (e instanceof ApiError && e.code === "network" && ctrl.signal.aborted) return;
          setError(e instanceof ApiError ? e.message : "Prediction failed.");
        })
        .finally(() => id === seq.current && setLoading(false));
    }, 220);
    return () => clearTimeout(t);
  }, [features, schema, cutoff]);

  const byName = useMemo(() => Object.fromEntries((schema?.features ?? []).map((f) => [f.name, f])), [schema]);
  const visible = useMemo(() => {
    if (!schema) return [];
    const informative = schema.features.filter((f) => f.kind === "categorical" || (f.p01 ?? 0) !== (f.p99 ?? 0));
    return showAll ? schema.features : informative.filter((f) => KEY_FEATURES.includes(f.name));
  }, [schema, showAll]);
  const groups = useMemo(() => {
    const g: Record<string, SchemaFeature[]> = {};
    for (const f of visible) (g[f.group] ??= []).push(f);
    return Object.entries(g).sort(([a], [b]) => Object.keys(FEATURE_GROUP_LABELS).indexOf(b) - Object.keys(FEATURE_GROUP_LABELS).indexOf(a));
  }, [visible]);

  const sample = samples.find((s) => s.id === sampleId);
  const set = (name: string, v: FeatureValue) => setFeatures((prev) => ({ ...prev, [name]: v }));
  const loadSample = (id: string) => {
    if (!schema) return;
    setSampleId(id);
    const s = samples.find((x) => x.id === id);
    setFeatures(s ? { ...baselineOf(schema), ...s.features } : baselineOf(schema));
    setRevealed(false);
  };
  const actions = useMemo(() => {
    if (!prediction || !schema) return [];
    const out = new Set<string>();
    for (const f of prediction.factors) {
      if (f.impact <= 0) continue;
      const a = ACTIONS.find((x) => x.test(f.name));
      if (a) out.add(a.text);
    }
    return [...out];
  }, [prediction, schema]);

  return (
    <>
      <PageHeader eyebrow="Live predictor" title="Predict a student's risk — and see why">
        Load a real student from the unseen {schema?.test_presentation ?? "test"} semester (the model never saw them), or build a what-if
        scenario. The deployed {schema?.model ?? "model"} returns a risk score and the factors that move it.
      </PageHeader>
      <Container className="mt-8">
        {error && (
          <div className="mb-6">
            <Callout tone="warn" title="Something went wrong">{error}</Callout>
          </div>
        )}
        <div className="grid gap-6 lg:grid-cols-[1fr_400px]">
          <div className="space-y-6">
            <Card>
              <div className="flex flex-col gap-4">
                <div>
                  <p className="mb-2 text-sm font-medium text-ink">Prediction day</p>
                  {cutoff !== null && cutoffs.length > 0 && (
                    <Segmented label="Prediction day" options={cutoffs} value={cutoff} onChange={setCutoff} format={(d) => `Day ${d}`} />
                  )}
                  <p className="mt-2 text-xs text-muted">Uses only information that existed on that day of the course.</p>
                </div>
                <div>
                  <label htmlFor="sample" className="mb-2 block text-sm font-medium text-ink">Start from</label>
                  <div className="flex flex-wrap gap-2">
                    <select
                      id="sample"
                      value={sampleId}
                      onChange={(e) => loadSample(e.target.value)}
                      className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-2.5 py-2 text-sm text-ink"
                    >
                      <option value="">Typical student (all values at the median)</option>
                      {samples.map((s) => (
                        <option key={s.id} value={s.id}>Real student {s.id}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      className={buttonClass.secondary}
                      onClick={() => samples.length && loadSample(samples[Math.floor(Math.random() * samples.length)]!.id)}
                    >
                      <Dice5 className="h-4 w-4" aria-hidden /> Random
                    </button>
                    <button type="button" className={buttonClass.secondary} onClick={() => loadSample(sampleId)} title="Undo your edits">
                      <RotateCcw className="h-4 w-4" aria-hidden /> Reset
                    </button>
                  </div>
                </div>
              </div>
            </Card>

            {groups.map(([group, feats]) => (
              <Card key={group}>
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="font-semibold text-ink">{FEATURE_GROUP_LABELS[group] ?? group}</h2>
                  {CONTEXT_GROUPS.has(group) && <Pill>context</Pill>}
                </div>
                <div className="grid gap-5 sm:grid-cols-2">
                  {feats.map((f) =>
                    f.kind === "numeric" ? (
                      <NumberField key={f.name} f={f} value={features[f.name] ?? null} onChange={(v) => set(f.name, v)} />
                    ) : (
                      <SelectField key={f.name} f={f} value={features[f.name] ?? null} onChange={(v) => set(f.name, v)} />
                    ),
                  )}
                </div>
              </Card>
            ))}
            {schema && (
              <button type="button" className={buttonClass.ghost} onClick={() => setShowAll((v) => !v)}>
                {showAll ? "Show key features only" : `Show all ${schema.features.length} features`}
              </button>
            )}
            <p className="text-xs leading-relaxed text-muted">
              What-if edits change one value at a time; related features (for example clicks relative to the class median) are not
              recalculated, so treat large edits as illustrative.
            </p>
          </div>

          <div className="lg:sticky lg:top-20 lg:self-start">
            <Card aria-live="polite">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-ink-2">Predicted risk of failing or withdrawing</p>
                {loading && <Spinner className="text-ink-2" />}
              </div>
              {prediction ? (
                <>
                  <p className={cx("mt-2 text-6xl font-semibold tracking-tight text-ink transition-opacity", loading && "opacity-60")}>
                    {formatPct(prediction.probability, 0)}
                  </p>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <RiskBadge band={prediction.band} size="lg" />
                    <span className="text-sm text-ink-2">
                      {prediction.at_risk ? "Flagged for early intervention" : "Below the intervention threshold"}
                    </span>
                  </div>
                  <div className="mt-5">
                    <RiskMeter probability={prediction.probability} threshold={prediction.threshold} />
                  </div>

                  <h3 className="mt-6 mb-3 text-sm font-semibold text-ink">Why: change in risk vs a typical student</h3>
                  {prediction.factors.length ? (
                    <DivergingBars
                      items={prediction.factors.map((f) => ({
                        label: f.label,
                        value: f.impact,
                        detail: `${fmtValue(byName[f.name], f.value)} (typical: ${fmtValue(byName[f.name], f.baseline)})${CONTEXT_GROUPS.has(byName[f.name]?.group ?? "") ? " · context" : ""}`,
                      }))}
                    />
                  ) : (
                    <p className="text-sm text-ink-2">This student matches the typical profile on every feature.</p>
                  )}

                  {actions.length > 0 && (
                    <div className="mt-6 rounded-xl bg-surface-2 p-4">
                      <p className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink">
                        <Lightbulb className="h-4 w-4" aria-hidden /> Suggested next steps
                      </p>
                      <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
                        {actions.map((a) => <li key={a}>{a}</li>)}
                      </ul>
                    </div>
                  )}

                  {sample && (
                    <div className="mt-6 border-t border-line pt-4">
                      {revealed ? (
                        <p className="flex items-start gap-2 text-sm text-ink">
                          <UserRound className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                          <span>
                            Actual final result: <strong>{sample.final_result}</strong>.{" "}
                            {(sample.at_risk === 1) === prediction.at_risk ? "The model's flag was correct." : "The model's flag was wrong for this student."}
                            {Object.keys(features).some((k) => features[k] !== sample.features[k]) && " (You have edited this student's values.)"}
                          </span>
                        </p>
                      ) : (
                        <button type="button" className={buttonClass.secondary} onClick={() => setRevealed(true)}>
                          <Eye className="h-4 w-4" aria-hidden /> Reveal actual outcome
                        </button>
                      )}
                    </div>
                  )}
                  <p className="mt-4 text-xs text-muted">
                    {prediction.model} · day {prediction.cutoff} · threshold {formatPct(prediction.threshold)} · model v{prediction.model_version}
                  </p>
                </>
              ) : (
                <div className="flex h-40 items-center justify-center text-ink-2">{error ? "Unavailable" : <Spinner />}</div>
              )}
            </Card>
          </div>
        </div>
      </Container>
    </>
  );
}
