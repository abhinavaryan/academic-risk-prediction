"use client";

import { ChartCard, DataTable, Legend, SERIES } from "@/components/charts/common";
import { HBarChart, LineSeriesChart, SegmentedBar } from "@/components/charts/charts";
import { Card, Container, PageHeader, SectionHeading, StatTile } from "@/components/ui";
import { eda, overview } from "@/lib/data";
import { compact, formatPct, int } from "@/lib/format";
import type { RateRow } from "@/lib/types";

// Colour follows the outcome everywhere on the site (fixed categorical order).
export const OUTCOME_COLORS: Record<string, string> = {
  Distinction: SERIES[0]!,
  Pass: SERIES[1]!,
  Fail: SERIES[2]!,
  Withdrawn: SERIES[3]!,
};

const IMD_ORDER = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown"];
const EDU_ORDER = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"];

function sortBy(rows: RateRow[], order: string[]) {
  return [...rows].sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group));
}

function RateCard({ title, rows, subtitle }: { title: string; rows: RateRow[]; subtitle?: string }) {
  return (
    <ChartCard
      title={title}
      subtitle={subtitle}
      table={{ columns: ["Group", "Registrations", "At-risk rate"], rows: rows.map((r) => [r.group, int(r.n), formatPct(r.at_risk_rate)]) }}
    >
      <HBarChart
        data={rows.map((r) => ({ label: r.group, value: r.at_risk_rate, note: `${int(r.n)} registrations` }))}
        format={(v) => formatPct(v, 0)}
        domain={[0, 1]}
        valueLabel="At-risk rate"
        rowHeight={30}
      />
    </ChartCard>
  );
}

export function DatasetView() {
  const d = overview.dataset;
  const total = eda.outcomes.reduce((s, o) => s + o.n, 0);
  const weeks = eda.engagement_by_week;
  const outcomes = eda.outcomes.map((o) => o.outcome);
  return (
    <>
      <PageHeader eyebrow="Dataset" title="Open University Learning Analytics Dataset (OULAD)">
        Anonymised records of {int(d.students)} students across {d.modules} modules and {d.module_presentations} course runs (2013–2014),
        including every click in the virtual learning environment. Published by the Open University under CC BY 4.0
        (Kuzilek, Hlosta &amp; Zdrahal, <em>Scientific Data</em> 4, 170171, 2017).
      </PageHeader>

      <Container className="mt-10 space-y-12">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile label="Registrations" value={int(d.registrations)} hint="studentInfo.csv" />
          <StatTile label="LMS activity records" value={compact(d.vle_interactions)} hint={`${compact(d.vle_clicks)} clicks · studentVle.csv`} />
          <StatTile label="Assessment submissions" value={int(d.assessment_submissions)} hint={`${d.assessments} assessments`} />
          <StatTile label="Learning resources" value={int(d.vle_resources)} hint="vle.csv" />
        </div>

        <Card>
          <h2 className="text-base font-semibold text-ink">Final results of all {int(total)} registrations</h2>
          <p className="mt-1 text-sm text-ink-2">
            We call a registration <strong>at risk</strong> if it ended in Fail or Withdrawn: {formatPct(eda.at_risk_rate)} of the dataset.
          </p>
          <div className="mt-5">
            <SegmentedBar parts={eda.outcomes.map((o) => ({ label: o.outcome, value: o.n, color: OUTCOME_COLORS[o.outcome]!, dark: o.outcome === "Withdrawn" }))} />
          </div>
          <div className="mt-3">
            <Legend items={eda.outcomes.map((o) => ({ label: `${o.outcome} · ${int(o.n)}`, color: OUTCOME_COLORS[o.outcome]! }))} />
          </div>
        </Card>

        <section>
          <SectionHeading eyebrow="Engagement" title="Students who later fail or withdraw are less active from the very first weeks">
            Average LMS clicks per registered student in each course week, split by the final result they eventually got.
          </SectionHeading>
          <ChartCard
            title="Average weekly LMS clicks per student, by final result"
            legend={outcomes.map((o) => ({ label: o, color: OUTCOME_COLORS[o]!, kind: "line" as const }))}
            table={{ columns: ["Week", ...outcomes], rows: weeks.map((w) => [w.week, ...outcomes.map((o) => (w[o] ?? 0).toFixed(1))]) }}
            footer="Week 0 starts on the first day of the course; negative weeks are pre-course access."
          >
            <LineSeriesChart
              data={weeks}
              xKey="week"
              xLabel="Week"
              series={outcomes.map((o) => ({ key: o, label: o, color: OUTCOME_COLORS[o]! }))}
              yFormat={(v) => v.toFixed(0)}
              height={320}
              reference={{ x: overview.primary_cutoff / 7, label: `day ${overview.primary_cutoff}` }}
            />
          </ChartCard>
        </section>

        <section>
          <SectionHeading eyebrow="Who is at risk" title="At-risk rates differ across student groups">
            These patterns are why demographic attributes are audited for fairness on the Models page; the model is checked so that it
            does not simply learn to flag disadvantaged groups.
          </SectionHeading>
          <div className="grid gap-4 lg:grid-cols-2">
            <RateCard title="By deprivation (IMD) band" subtitle="0–10% = most deprived areas" rows={sortBy(eda.by_imd_band, IMD_ORDER)} />
            <RateCard title="By highest prior education" rows={sortBy(eda.by_highest_education, EDU_ORDER)} />
            <RateCard title="By module" rows={eda.by_module} />
            <RateCard title="By age band" rows={eda.by_age_band} />
          </div>
        </section>

        <section>
          <SectionHeading eyebrow="Assessments" title="Average assessment score over the course, by final result" />
          <ChartCard
            title="Spread of each student's mean assessment score"
            subtitle="Thin bar: 10th–90th percentile · thick bar: 25th–75th · tick: median."
            table={{
              columns: ["Result", "Students", "P10", "P25", "Median", "P75", "P90"],
              rows: eda.score_by_outcome.map((s) => [s.outcome, int(s.n), s.p10.toFixed(0), s.p25.toFixed(0), s.median.toFixed(0), s.p75.toFixed(0), s.p90.toFixed(0)]),
            }}
          >
            <ul className="space-y-4">
              {eda.score_by_outcome.map((s) => (
                <li key={s.outcome} className="grid grid-cols-[7rem_1fr] items-center gap-3 text-sm">
                  <span className="text-ink-2">{s.outcome}</span>
                  <div className="relative h-6" role="img" aria-label={`${s.outcome}: median ${s.median.toFixed(0)}, middle half ${s.p25.toFixed(0)} to ${s.p75.toFixed(0)}`}>
                    <div className="absolute inset-x-0 top-1/2 h-px bg-[var(--grid)]" />
                    <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full opacity-50" style={{ left: `${s.p10}%`, width: `${s.p90 - s.p10}%`, background: OUTCOME_COLORS[s.outcome] }} />
                    <div className="absolute top-1/2 h-3 -translate-y-1/2 rounded" style={{ left: `${s.p25}%`, width: `${s.p75 - s.p25}%`, background: OUTCOME_COLORS[s.outcome] }} />
                    <div className="absolute top-0 h-6 w-0.5 rounded bg-[var(--ink)]" style={{ left: `${s.median}%` }} />
                  </div>
                </li>
              ))}
              <li className="grid grid-cols-[7rem_1fr] text-xs text-muted">
                <span />
                <span className="flex justify-between"><span>0</span><span>50</span><span>100</span></span>
              </li>
            </ul>
          </ChartCard>
        </section>

        <section>
          <SectionHeading eyebrow="Structure" title="Course runs in the dataset">
            Course runs starting in February are coded B, October starts J. Models train on 2013B, 2013J and 2014B and are evaluated on{" "}
            {overview.split.test_presentation}, the most recent semester.
          </SectionHeading>
          <DataTable
            spec={{
              columns: ["Module", "Presentation", "Length (days)", "Registrations", "Role"],
              rows: eda.presentations
                .slice()
                .sort((a, b) => a.code_module.localeCompare(b.code_module) || a.code_presentation.localeCompare(b.code_presentation))
                .map((p) => [p.code_module, p.code_presentation, p.length_days, int(p.registrations), p.code_presentation === overview.split.test_presentation ? "Test" : "Train"]),
            }}
          />
        </section>
      </Container>
    </>
  );
}
