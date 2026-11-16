"use client";

import { ArrowDownUp, Award, Rocket } from "lucide-react";
import { useMemo, useState } from "react";
import { ChartCard, SERIES } from "@/components/charts/common";
import { ConfusionMatrix, HBarChart, LineSeriesChart, UnitCurveChart } from "@/components/charts/charts";
import { Callout, Card, Container, PageHeader, Pill, SectionHeading, Segmented, cx } from "@/components/ui";
import { ablation, curves, fairness, importance, leaderboard, overview, timeline } from "@/lib/data";
import { fixed, formatPct, int } from "@/lib/format";
import type { ModelRecord } from "@/lib/types";

type SortKey = "rank" | "cv_auc" | "test_auc" | "f1" | "recall" | "precision" | "accuracy" | "mcc";
const SORTERS: Record<SortKey, (m: ModelRecord) => number> = {
  rank: (m) => -m.rank,
  cv_auc: (m) => m.cv.roc_auc,
  test_auc: (m) => m.test.roc_auc,
  f1: (m) => m.test.f1,
  recall: (m) => m.test.recall,
  precision: (m) => m.test.precision,
  accuracy: (m) => m.test.accuracy,
  mcc: (m) => m.test.mcc,
};

function pValue(p: number | undefined) {
  if (p === undefined) return "—";
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}

function Leaderboard() {
  const [sort, setSort] = useState<SortKey>("rank");
  const rows = useMemo(() => [...leaderboard.models].sort((a, b) => SORTERS[sort](b) - SORTERS[sort](a)), [sort]);
  const head = (key: SortKey, label: string, title?: string) => (
    <th scope="col" className="px-3 py-2.5 text-right font-medium" aria-sort={sort === key ? "descending" : "none"}>
      <button type="button" onClick={() => setSort(key)} title={title} className={cx("inline-flex items-center gap-1 hover:text-ink", sort === key && "text-ink")}>
        {label}
        <ArrowDownUp className="h-3 w-3" aria-hidden />
      </button>
    </th>
  );
  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
      <table className="w-full min-w-[960px] text-sm">
        <caption className="sr-only">Model leaderboard at day {leaderboard.cutoff}</caption>
        <thead className="bg-surface-2 text-ink-2">
          <tr>
            {head("rank", "#", "Rank by cross-validated ROC-AUC")}
            <th scope="col" className="px-3 py-2.5 text-left font-medium">Model</th>
            {head("cv_auc", "CV ROC-AUC", "Mean of 5 student-grouped folds ± standard deviation")}
            {head("test_auc", "Test ROC-AUC", "Unseen semester, with 95% bootstrap CI")}
            {head("f1", "F1")}
            {head("recall", "Recall")}
            {head("precision", "Precision")}
            {head("accuracy", "Accuracy")}
            {head("mcc", "MCC", "Matthews correlation coefficient")}
            <th scope="col" className="px-3 py-2.5 text-right font-medium" title="McNemar test against the champion on the test set">p vs best</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((m) => {
            const isChamp = m.name === leaderboard.champion;
            const isDeployed = m.name === leaderboard.deployed;
            const ci = m.ci95?.roc_auc;
            return (
              <tr key={m.name} className={cx("border-t border-line", isDeployed && "bg-accent-wash/50")}>
                <td className="px-3 py-2.5 text-right text-ink-2">{m.rank}</td>
                <td className="px-3 py-2.5 text-left">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-medium text-ink">{m.name}</span>
                    {isChamp && (
                      <Pill tone="accent">
                        <Award className="h-3 w-3" aria-hidden /> Best CV
                      </Pill>
                    )}
                    {isDeployed && (
                      <Pill tone="accent">
                        <Rocket className="h-3 w-3" aria-hidden /> Deployed
                      </Pill>
                    )}
                  </div>
                  <span className="text-xs text-muted">{m.family}</span>
                </td>
                <td className="px-3 py-2.5 text-right text-ink">
                  {fixed(m.cv.roc_auc)} <span className="text-xs text-muted">±{fixed(m.cv.roc_auc_std)}</span>
                </td>
                <td className="px-3 py-2.5 text-right text-ink">
                  {fixed(m.test.roc_auc)}
                  {ci && <span className="block text-xs text-muted">{fixed(ci[0])}–{fixed(ci[1])}</span>}
                </td>
                <td className="px-3 py-2.5 text-right text-ink">{fixed(m.test.f1)}</td>
                <td className="px-3 py-2.5 text-right text-ink">{formatPct(m.test.recall)}</td>
                <td className="px-3 py-2.5 text-right text-ink">{formatPct(m.test.precision)}</td>
                <td className="px-3 py-2.5 text-right text-ink">{formatPct(m.test.accuracy)}</td>
                <td className="px-3 py-2.5 text-right text-ink">{fixed(m.test.mcc)}</td>
                <td className="px-3 py-2.5 text-right text-ink-2">{isChamp ? "—" : pValue(m.mcnemar_vs_champion?.p_value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ModelsView() {
  const deployed = leaderboard.models.find((m) => m.name === leaderboard.deployed)!;
  const champion = leaderboard.models.find((m) => m.name === leaderboard.champion)!;
  const ranked = [...leaderboard.models].sort((a, b) => a.rank - b.rank);
  const rocModels = Array.from(new Set([champion.name, deployed.name, ...ranked.map((m) => m.name)]))
    .filter((n) => n !== "Baseline (majority)" && curves[n])
    .slice(0, 4);
  const [metric, setMetric] = useState<"roc_auc" | "f1" | "recall">("roc_auc");
  const metricLabel = { roc_auc: "ROC-AUC", f1: "F1-score", recall: "Recall" }[metric];
  const tlData = timeline.cutoffs.map((c) => ({
    day: c.cutoff,
    ...Object.fromEntries(timeline.models.map((m) => [m, c.models[m]?.[metric] ?? null])),
  }));
  const perm = importance.permutation.slice(0, 15);
  const shap = importance.shap?.mean_abs.slice(0, 15);
  const tunedNames = Object.keys(leaderboard.tuning);

  return (
    <>
      <PageHeader eyebrow="Models" title={`${leaderboard.models.length} models, one honest comparison`}>
        Every model is trained on the {overview.split.train_presentations.join(", ")} semesters with {overview.split.cv_folds}-fold
        cross-validation grouped by student, so the same student never appears in both training and validation folds. The decision
        threshold is tuned on out-of-fold predictions. Only then is each model scored <strong>once</strong> on the {overview.split.test_presentation}{" "}
        semester ({int(overview.split.n_test)} registrations) that played no part in training or selection. Prediction day: {leaderboard.cutoff}.
      </PageHeader>

      <Container className="mt-10 space-y-12">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <p className="flex items-center gap-2 text-sm font-medium text-ink-2"><Award className="h-4 w-4" aria-hidden /> Best by cross-validation</p>
            <p className="mt-2 text-xl font-semibold text-ink">{champion.name}</p>
            <p className="mt-1 text-sm text-ink-2">
              CV ROC-AUC {fixed(champion.cv.roc_auc)} · test ROC-AUC {fixed(champion.test.roc_auc)} · test F1 {fixed(champion.test.f1)}
            </p>
          </Card>
          <Card className="border-accent/30">
            <p className="flex items-center gap-2 text-sm font-medium text-ink-2"><Rocket className="h-4 w-4" aria-hidden /> Deployed in this app</p>
            <p className="mt-2 text-xl font-semibold text-ink">{deployed.name}</p>
            <p className="mt-1 text-sm text-ink-2">
              CV ROC-AUC {fixed(deployed.cv.roc_auc)} · test ROC-AUC {fixed(deployed.test.roc_auc)} · test F1 {fixed(deployed.test.f1)}
            </p>
            <p className="mt-2 text-xs leading-relaxed text-muted">{overview.deployed.reason}</p>
          </Card>
        </div>

        <section>
          <SectionHeading eyebrow="Leaderboard" title="All models on the unseen test semester">
            Sorted by cross-validated ROC-AUC (the pre-registered selection metric). Click a column to sort. The last column is a McNemar
            test against the best model: p &lt; 0.05 means the two models make significantly different mistakes.
          </SectionHeading>
          <Leaderboard />
        </section>

        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            title="Test ROC-AUC with 95% confidence intervals"
            subtitle="Bootstrap over the test semester (1,000 resamples). Deployed and best-CV models highlighted."
            table={{ columns: ["Model", "Test ROC-AUC", "95% CI"], rows: ranked.map((m) => [m.name, fixed(m.test.roc_auc), m.ci95?.roc_auc ? `${fixed(m.ci95.roc_auc[0])}–${fixed(m.ci95.roc_auc[1])}` : "—"]) }}
          >
            <HBarChart
              data={ranked.map((m) => ({ label: m.name, value: m.test.roc_auc, ci: m.ci95?.roc_auc, highlight: m.name === deployed.name || m.name === champion.name }))}
              domain={[0.5, 1]}
              valueLabel="Test ROC-AUC"
            />
          </ChartCard>
          <ChartCard
            title="ROC curves"
            subtitle="True-positive rate vs false-positive rate on the test semester; the diagonal is random guessing."
            legend={rocModels.map((n, i) => ({ label: n, color: SERIES[i]!, kind: "line" }))}
            table={{ columns: ["Model", "ROC-AUC"], rows: rocModels.map((n) => [n, fixed(leaderboard.models.find((m) => m.name === n)?.test.roc_auc)]) }}
          >
            <UnitCurveChart
              curves={rocModels.map((n, i) => ({ name: n, color: SERIES[i]!, points: curves[n]!.roc }))}
              xLabel="False-positive rate"
              yLabel="True-positive rate"
              height={340}
            />
          </ChartCard>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            title={`Confusion matrix — ${deployed.name}`}
            subtitle={`Test semester at the tuned threshold of ${formatPct(deployed.threshold)} risk.`}
            table={{ columns: ["", "Predicted at risk", "Predicted not at risk"], rows: [["Actually at risk", deployed.test.tp, deployed.test.fn], ["Actually not at risk", deployed.test.fp, deployed.test.tn]] }}
          >
            <ConfusionMatrix tp={deployed.test.tp} fp={deployed.test.fp} tn={deployed.test.tn} fn={deployed.test.fn} />
          </ChartCard>
          <ChartCard
            title="Calibration: does a 70% risk mean 70%?"
            subtitle="Mean predicted risk vs the observed at-risk share in ten probability bins (test semester)."
            legend={[deployed.name, champion.name].filter((v, i, a) => a.indexOf(v) === i).map((n, i) => ({ label: n, color: SERIES[i]!, kind: "line" }))}
            table={{ columns: ["Model", "Mean predicted", "Observed", "Students"], rows: [deployed.name, champion.name].filter((v, i, a) => a.indexOf(v) === i).flatMap((n) => (curves[n]?.calibration ?? []).map((c) => [n, fixed(c[0], 2), fixed(c[1], 2), int(c[2])])) }}
            footer={`Brier score (lower is better): ${deployed.name} ${fixed(deployed.test.brier)}${champion.name !== deployed.name ? `, ${champion.name} ${fixed(champion.test.brier)}` : ""}.`}
          >
            <UnitCurveChart
              curves={[deployed.name, champion.name]
                .filter((v, i, a) => a.indexOf(v) === i)
                .map((n, i) => ({ name: n, color: SERIES[i]!, points: (curves[n]?.calibration ?? []).map((c) => [c[0], c[1]] as [number, number]) }))}
              xLabel="Mean predicted risk"
              yLabel="Observed at-risk share"
              height={300}
            />
          </ChartCard>
        </div>

        <section>
          <SectionHeading eyebrow="Early warning" title="How early can we predict?">
            The same models re-trained using only data available on each day. Earlier predictions are less accurate but leave more time
            to help, and some students withdraw before later prediction days.
          </SectionHeading>
          <ChartCard
            title={`${metricLabel} on the test semester by prediction day`}
            legend={timeline.models.map((m, i) => ({ label: m, color: SERIES[i]!, kind: "line" }))}
            table={{
              columns: ["Day", "Students still enrolled", "At-risk among them", ...timeline.models],
              rows: timeline.cutoffs.map((c) => [c.cutoff, int(c.n_test), formatPct(c.at_risk_rate_test), ...timeline.models.map((m) => fixed(c.models[m]?.[metric]))]),
            }}
          >
            <div className="no-print mb-4">
              <Segmented<"roc_auc" | "f1" | "recall"> label="Metric" options={["roc_auc", "f1", "recall"]} value={metric} onChange={setMetric} format={(v) => ({ roc_auc: "ROC-AUC", f1: "F1", recall: "Recall" })[v]} />
            </div>
            <LineSeriesChart
              data={tlData}
              xKey="day"
              xLabel="Day"
              series={timeline.models.map((m, i) => ({ key: m, label: m, color: SERIES[i]! }))}
              yFormat={(v) => v.toFixed(2)}
              height={320}
            />
          </ChartCard>
        </section>

        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            title="Ablation: what each feature group adds"
            subtitle={`${ablation.model}, test ROC-AUC at day ${leaderboard.cutoff}. "Demographics only" mirrors a profile-only model.`}
            table={{ columns: ["Feature set", "Features", "ROC-AUC", "F1", "Recall"], rows: ablation.variants.map((v) => [v.variant, v.n_features, fixed(v.roc_auc), fixed(v.f1), formatPct(v.recall)]) }}
          >
            <HBarChart data={ablation.variants.map((v) => ({ label: v.variant, value: v.roc_auc, note: `${v.n_features} features` }))} domain={[0.5, 1]} valueLabel="Test ROC-AUC" />
          </ChartCard>
          <ChartCard
            title="What drives the deployed model"
            subtitle="Permutation importance: drop in test ROC-AUC when a feature's values are shuffled (top 15)."
            table={{ columns: ["Feature", "AUC drop", "±"], rows: importance.permutation.map((p) => [p.label, fixed(p.mean, 4), fixed(p.std, 4)]) }}
          >
            <HBarChart data={perm.map((p) => ({ label: p.label, value: Math.max(0, p.mean) }))} format={(v) => v.toFixed(3)} valueLabel="AUC drop" rowHeight={26} />
          </ChartCard>
        </div>

        {shap && (
          <ChartCard
            title={`SHAP feature attribution — ${importance.shap_model}`}
            subtitle="Mean absolute TreeSHAP contribution to the log-odds on the test semester (top 15); a cross-check from a different model family."
            table={{ columns: ["Feature", "Mean |SHAP|"], rows: importance.shap!.mean_abs.map((s) => [s.label, fixed(s.value, 4)]) }}
          >
            <HBarChart data={shap.map((s) => ({ label: s.label, value: s.value }))} format={(v) => v.toFixed(3)} valueLabel="Mean |SHAP|" rowHeight={26} />
          </ChartCard>
        )}

        {tunedNames.length > 0 && (
          <ChartCard
            title="Bayesian hyper-parameter tuning (Optuna)"
            subtitle="Best 3-fold grouped CV ROC-AUC found so far, by trial."
            legend={tunedNames.map((n, i) => ({ label: n, color: SERIES[i]!, kind: "line" }))}
            table={{ columns: ["Model", "Trials", "Best CV AUC", "Best parameters"], rows: tunedNames.map((n) => [n, leaderboard.tuning[n]!.n_trials, fixed(Math.max(...leaderboard.tuning[n]!.history.map((h) => h.best)), 4), Object.entries(leaderboard.tuning[n]!.best_params).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toFixed(4) : v}`).join(", ")]) }}
          >
            <LineSeriesChart
              data={Array.from({ length: Math.max(...tunedNames.map((n) => leaderboard.tuning[n]!.history.length)) }, (_, t) => ({
                trial: t + 1,
                ...Object.fromEntries(tunedNames.map((n) => [n, leaderboard.tuning[n]!.history[t]?.best ?? null])),
              }))}
              xKey="trial"
              xLabel="Trial"
              series={tunedNames.map((n, i) => ({ key: n, label: n, color: SERIES[i]! }))}
              yFormat={(v) => v.toFixed(3)}
            />
          </ChartCard>
        )}

        <section>
          <SectionHeading eyebrow="Responsible AI" title="Fairness audit of the deployed model">
            For each group we compare how often the model flags students (flag rate) and how many truly at-risk students it catches (recall
            / true-positive rate). Large gaps would mean some groups are systematically missed or over-flagged.
          </SectionHeading>
          <div className="grid gap-4 lg:grid-cols-2">
            {fairness.attributes.map((a) => (
              <Card key={a.attribute}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="font-semibold text-ink">{a.label}</h3>
                  <p className="text-xs text-ink-2">
                    recall gap {fixed(a.gaps.tpr_gap, 3)} · flag-rate gap {fixed(a.gaps.positive_rate_gap, 3)}
                    {a.gaps_without_protected && <> · without protected attributes: recall gap {fixed(a.gaps_without_protected.tpr_gap, 3)}</>}
                  </p>
                </div>
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full min-w-[520px] text-sm [&_td]:px-2 [&_th]:px-2">
                    <thead className="text-ink-2">
                      <tr>
                        <th scope="col" className="py-1.5 text-left font-medium">Group</th>
                        <th scope="col" className="py-1.5 text-right font-medium">Students</th>
                        <th scope="col" className="py-1.5 text-right font-medium">At-risk rate</th>
                        <th scope="col" className="py-1.5 text-right font-medium">Flag rate</th>
                        <th scope="col" className="py-1.5 text-right font-medium">Recall</th>
                        <th scope="col" className="py-1.5 text-right font-medium">False alarms</th>
                      </tr>
                    </thead>
                    <tbody className="tabular">
                      {a.groups.map((g) => (
                        <tr key={g.group} className="border-t border-line">
                          <td className="py-1.5 text-ink">{g.group}</td>
                          <td className="py-1.5 text-right text-ink-2">{int(g.n)}</td>
                          <td className="py-1.5 text-right text-ink">{formatPct(g.base_rate)}</td>
                          <td className="py-1.5 text-right text-ink">{formatPct(g.positive_rate)}</td>
                          <td className="py-1.5 text-right text-ink">{formatPct(g.tpr)}</td>
                          <td className="py-1.5 text-right text-ink">{formatPct(g.fpr)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            ))}
          </div>
          <div className="mt-4">
            <Callout title="How to read this">
              Flag rates track each group&apos;s real at-risk rate; that is expected. The concern is recall: if one group&apos;s recall is much
              lower, at-risk students in that group are being missed. The &ldquo;All except protected attributes&rdquo; ablation shows the
              accuracy cost of removing gender, disability, age, deprivation and region from the model entirely.
            </Callout>
          </div>
        </section>
      </Container>
    </>
  );
}
