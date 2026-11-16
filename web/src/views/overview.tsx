"use client";

import { ArrowRight, BarChart3, Brain, Database, FileUp, ShieldCheck, Target } from "lucide-react";
import Link from "next/link";
import { ChartCard, SERIES } from "@/components/charts/common";
import { LineSeriesChart } from "@/components/charts/charts";
import { Card, Container, SectionHeading, StatTile, buttonClass } from "@/components/ui";
import { site } from "@/config/site";
import { eda, overview, timeline } from "@/lib/data";
import { compact, fixed, formatPct, int } from "@/lib/format";

export function OverviewView() {
  const o = overview;
  const ew = o.early_warning;
  const ci = ew.ci95?.recall;
  const deployedBase = timeline.deployed;
  const tl = timeline.cutoffs.map((c) => ({
    day: c.cutoff,
    auc: c.models[deployedBase]?.roc_auc ?? null,
    recall: c.models[deployedBase]?.recall ?? null,
  }));
  const wdAtPrimary = eda.withdrawal_curve.find((w) => w.day === o.primary_cutoff)?.share;
  const caught = ew.tp;
  const remaining = ew.at_risk_remaining_test;

  return (
    <>
      <section className="border-b border-line bg-surface">
        <Container className="grid gap-10 py-12 sm:py-16 lg:grid-cols-[1.25fr_1fr] lg:items-center">
          <div>
            <p className="mb-4 text-sm font-medium text-accent-ink">Learning analytics · Early-warning system</p>
            <h1 className="text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
              Spot at-risk students weeks before they fail or withdraw.
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-2">
              {site.name} learns from {int(o.dataset.registrations)} real course registrations ({int(o.dataset.students)} students,{" "}
              {compact(o.dataset.vle_interactions)} LMS activity records) and is tested on a later semester the models never saw.
              Bring your own college&apos;s records and it trains on those too.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/predict/" className={buttonClass.primary}>
                Try the risk predictor <ArrowRight className="h-4 w-4" aria-hidden />
              </Link>
              <Link href="/train/" className={buttonClass.secondary}>
                <FileUp className="h-4 w-4" aria-hidden /> Train on your data
              </Link>
            </div>
          </div>

          <Card className="bg-page">
            <p className="text-sm text-ink-2">At day {o.primary_cutoff} of the course, the deployed model flagged</p>
            <p className="mt-2 text-6xl font-semibold tracking-tight text-ink">{formatPct(ew.recall, 0)}</p>
            <p className="mt-2 text-base text-ink-2">
              of the students who went on to fail or withdraw ({int(caught)} of {int(remaining)} in the unseen {o.split.test_presentation}{" "}
              semester{ci ? `; 95% CI ${formatPct(ci[0], 0)}–${formatPct(ci[1], 0)}` : ""}).
            </p>
            <dl className="mt-6 grid grid-cols-2 gap-4 border-t border-line pt-4 text-sm">
              <div>
                <dt className="text-ink-2">Precision</dt>
                <dd className="mt-0.5 text-lg font-semibold text-ink">{formatPct(ew.precision, 0)}</dd>
                <dd className="text-xs text-muted">of flagged students really were at risk</dd>
              </div>
              <div>
                <dt className="text-ink-2">ROC-AUC</dt>
                <dd className="mt-0.5 text-lg font-semibold text-ink">{fixed(ew.roc_auc, 3)}</dd>
                <dd className="text-xs text-muted">{o.deployed.name}</dd>
              </div>
            </dl>
          </Card>
        </Container>
      </section>

      <Container className="mt-10">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <StatTile label="Course registrations" value={int(o.dataset.registrations)} />
          <StatTile label="Unique students" value={int(o.dataset.students)} />
          <StatTile label="LMS activity records" value={compact(o.dataset.vle_interactions)} hint={`${compact(o.dataset.vle_clicks)} clicks`} />
          <StatTile label="Models compared" value={int(o.n_models)} />
          <StatTile label="Features per student" value={int(o.n_features)} />
          <StatTile label="Prediction days" value={int(o.cutoffs.length)} hint={`day ${o.cutoffs[0]} to ${o.cutoffs[o.cutoffs.length - 1]}`} />
        </div>
      </Container>

      <Container className="mt-16">
        <SectionHeading eyebrow="Why early matters" title="Most withdrawals happen early — and the signal is already there">
          {wdAtPrimary !== undefined && (
            <>
              By day {o.primary_cutoff}, {formatPct(wdAtPrimary, 0)} of all withdrawals in the dataset have already happened. Waiting for
              exam results is too late; engagement and early assessment data are available within weeks.
            </>
          )}
        </SectionHeading>
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            title="How accurate is the model on each day of the course?"
            subtitle={`${o.deployed.name}, evaluated on the unseen ${o.split.test_presentation} semester.`}
            legend={[
              { label: "ROC-AUC (ranking quality)", color: SERIES[0]!, kind: "line" },
              { label: "Recall (share of at-risk students flagged)", color: SERIES[1]!, kind: "line" },
            ]}
            table={{
              columns: ["Day", "ROC-AUC", "Recall"],
              rows: tl.map((r) => [r.day, fixed(r.auc, 3), formatPct(r.recall)]),
            }}
          >
            <LineSeriesChart
              data={tl}
              xKey="day"
              xLabel="Day"
              series={[
                { key: "auc", label: "ROC-AUC", color: SERIES[0]! },
                { key: "recall", label: "Recall", color: SERIES[1]! },
              ]}
              yFormat={(v) => v.toFixed(2)}
              yDomain={[0.5, 1]}
              reference={{ x: o.primary_cutoff, label: `day ${o.primary_cutoff}` }}
            />
          </ChartCard>
          <ChartCard
            title="Cumulative share of withdrawals by course day"
            subtitle="Every student whose final result was Withdrawn, by the day they unregistered."
            table={{ columns: ["Day", "Share of withdrawals"], rows: eda.withdrawal_curve.map((w) => [w.day, formatPct(w.share)]) }}
          >
            <LineSeriesChart
              data={eda.withdrawal_curve}
              xKey="day"
              xLabel="Day"
              series={[{ key: "share", label: "Withdrawn so far", color: SERIES[0]! }]}
              xTicks={[-60, 0, 60, 120, 180, 240]}
              yFormat={(v) => formatPct(v, 0)}
              yDomain={[0, 1]}
              reference={{ x: o.primary_cutoff, label: `day ${o.primary_cutoff}` }}
            />
          </ChartCard>
        </div>
      </Container>

      <Container className="mt-16">
        <SectionHeading eyebrow="How it works" title="From raw clickstream to an explainable risk score" />
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { icon: Database, title: "Real data", body: `All 7 OULAD tables: ${int(o.dataset.registrations)} registrations, ${compact(o.dataset.vle_interactions)} LMS records, ${int(o.dataset.assessment_submissions)} assessment submissions.` },
            { icon: Target, title: "Point-in-time features", body: `${o.n_features} features per student built only from what was known on the prediction day, so there is no peeking into the future.` },
            { icon: Brain, title: `${o.n_models} models compared`, body: `Student-grouped ${o.split.cv_folds}-fold cross-validation, Bayesian tuning, a stacked ensemble, then one final test on a later semester.` },
            { icon: ShieldCheck, title: "Explainable & audited", body: "Every prediction lists the factors behind it; the model is checked for fairness across gender, disability, age and deprivation." },
          ].map((s, i) => (
            <li key={s.title}>
              <Card className="h-full">
                <div className="mb-4 flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-wash text-accent-ink">
                    <s.icon className="h-5 w-5" aria-hidden />
                  </span>
                  <span className="text-sm font-medium text-muted">Step {i + 1}</span>
                </div>
                <h3 className="font-semibold text-ink">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-2">{s.body}</p>
              </Card>
            </li>
          ))}
        </ol>
      </Container>

      <Container className="mt-16">
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { href: "/models/", icon: BarChart3, title: "See the model comparison", body: "Leaderboard with confidence intervals, ROC curves, ablation, feature importance and a fairness audit." },
            { href: "/predict/", icon: Target, title: "Predict a student's risk", body: "Load a real student from the unseen semester, change their engagement and see the risk move." },
            { href: "/train/", icon: FileUp, title: "Use your own college data", body: "Upload a CSV or Excel sheet; models are trained and compared on it in seconds. Nothing is stored." },
          ].map((c) => (
            <Link key={c.href} href={c.href} className="group">
              <Card className="h-full transition group-hover:border-accent/40">
                <c.icon className="h-5 w-5 text-accent-ink" aria-hidden />
                <h3 className="mt-3 font-semibold text-ink">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-2">{c.body}</p>
                <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-accent-ink">
                  Open <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" aria-hidden />
                </span>
              </Card>
            </Link>
          ))}
        </div>
      </Container>
    </>
  );
}
