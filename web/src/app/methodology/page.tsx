import type { Metadata } from "next";
import { Card, Container, PageHeader } from "@/components/ui";
import { site } from "@/config/site";
import { leaderboard, overview } from "@/lib/data";
import { compact, int } from "@/lib/format";

export const metadata: Metadata = { title: "Methodology" };

function H({ children }: { children: React.ReactNode }) {
  return <h2 className="mt-12 mb-4 text-2xl font-semibold tracking-tight text-ink">{children}</h2>;
}
function P({ children }: { children: React.ReactNode }) {
  return <p className="mb-4 max-w-3xl leading-relaxed text-ink-2">{children}</p>;
}

const ARCH = [
  { title: "OULAD (UCI, SHA-256 verified)", body: "7 CSV tables" },
  { title: "Point-in-time features", body: "riskml.oulad.features" },
  { title: "Research pipeline", body: "CV · Optuna · bootstrap · SHAP · fairness" },
  { title: "Artifacts", body: "skops models + manifest + JSON reports" },
  { title: "FastAPI", body: "/api/predict · /api/custom/*" },
  { title: "Next.js UI", body: "static export on a CDN" },
];

export default function Page() {
  const o = overview;
  return (
    <>
      <PageHeader eyebrow="Methodology" title="How the study was designed, and why you can trust the numbers">
        Every metric on this site is produced by one reproducible command (<code className="font-mono text-base">make research</code>) on the
        public dataset; nothing is typed in by hand.
      </PageHeader>
      <Container>
        <H>1. Problem formulation</H>
        <P>
          Each <strong>registration</strong> (a student taking one module in one semester) is a sample. The label is <strong>at risk</strong> = final
          result Fail or Withdrawn (positive class), versus Pass or Distinction. A prediction is made on a given <strong>day</strong> of the course
          (day {o.cutoffs.join(", ")}). Students who already unregistered before that day are excluded: they are no longer a warning target.
        </P>

        <H>2. No peeking into the future</H>
        <P>
          Features for day D use only events dated on or before D: clicks in the virtual learning environment, assessments submitted by D, and
          information known at registration. Exams are excluded. An automated test appends future clicks and submissions to the data and
          checks that the features do not change. Cohort-relative features (e.g. clicks versus the class median) only use classmates&apos; data up
          to the same day.
        </P>

        <H>3. Validation protocol</H>
        <ul className="mb-4 max-w-3xl list-disc space-y-2 pl-6 leading-relaxed text-ink-2">
          <li>
            <strong>Temporal hold-out:</strong> train on {o.split.train_presentations.join(", ")} ({int(o.split.n_train)} registrations at day{" "}
            {o.primary_cutoff}); test once on {o.split.test_presentation} ({int(o.split.n_test)}), the most recent semester, as a real deployment would.
          </li>
          <li>
            <strong>Model selection:</strong> {o.split.cv_folds}-fold StratifiedGroupKFold on the training semesters, grouped by student so the same
            person never appears on both sides of a split. Selection metric: cross-validated ROC-AUC, fixed before seeing test results.
          </li>
          <li><strong>Threshold:</strong> the at-risk cut-off maximises F1 on out-of-fold predictions; the test set is never used to tune anything.</li>
          <li><strong>Tuning:</strong> Optuna (TPE) Bayesian optimisation for the four strongest model families, 3-fold grouped CV inside the training data.</li>
          <li><strong>Uncertainty:</strong> 95% percentile-bootstrap confidence intervals (1,000 resamples) for every test metric; McNemar&apos;s test against the best model.</li>
        </ul>

        <H>4. Models compared</H>
        <P>{leaderboard.models.map((m) => m.name).join(" · ")}.</P>
        <P>
          The model deployed in the live app is <strong>{o.deployed.name}</strong>. {o.deployed.reason} The best model by cross-validation was{" "}
          <strong>{o.champion.name}</strong>; both are shown side by side on the Models page.
        </P>

        <H>5. Metrics, in plain words</H>
        <div className="grid max-w-4xl gap-3 sm:grid-cols-2">
          {[
            ["Recall", "Of the students who really failed or withdrew, how many did we flag? Missing an at-risk student is the costliest error."],
            ["Precision", "Of the students we flagged, how many really were at risk? Low precision wastes counselling time."],
            ["F1-score", "Harmonic mean of precision and recall; one number that balances both."],
            ["ROC-AUC", "Probability that a random at-risk student gets a higher risk score than a random safe student (0.5 = coin flip)."],
            ["MCC", "Correlation between predictions and truth that stays honest when classes are imbalanced."],
            ["Brier score", "Mean squared error of the probabilities; checks that a 70% risk really means about 70%."],
          ].map(([t, d]) => (
            <Card key={t}>
              <p className="font-semibold text-ink">{t}</p>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{d}</p>
            </Card>
          ))}
        </div>

        <H>6. Explainability and fairness</H>
        <P>
          Each live prediction lists the factors that move the risk: the model is re-scored with one feature at a time replaced by the typical
          (median / most common) value, and the change is that feature&apos;s impact. Globally we report permutation importance and, as a cross-check,
          TreeSHAP from a LightGBM model. The fairness audit compares flag rates, recall and false-alarm rates across gender, disability, age band,
          deprivation band and prior education, and an ablation retrains the model without protected attributes.
        </P>

        <H>7. Bring-your-own-data training</H>
        <P>
          Uploaded files are parsed defensively (encoding and delimiter detection, Excel zip-bomb guard, no formula evaluation), profiled to find the
          result column, IDs and personal data, screened for columns that leak the answer, and compared across up to eight scikit-learn models with
          stratified cross-validation under a time budget. Rows with an unknown result are scored by the refit best model. Files are processed in
          memory and never stored.
        </P>

        <H>8. Limitations</H>
        <ul className="mb-4 max-w-3xl list-disc space-y-2 pl-6 leading-relaxed text-ink-2">
          <li>OULAD is from one UK distance-learning university (2013–14); campus colleges in other countries behave differently, which is why the custom-training mode exists.</li>
          <li>Assessment scores are assumed known on the submission day; real marking delays would make early predictions slightly harder.</li>
          <li>A risk score is a prompt for a human conversation, never an automatic decision about a student.</li>
        </ul>

        <H>9. Architecture</H>
        <ol className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {ARCH.map((a, i) => (
            <li key={a.title} className="relative rounded-xl border border-line bg-surface p-4">
              <span className="text-xs font-medium text-muted">{i + 1}</span>
              <p className="mt-1 text-sm font-semibold text-ink">{a.title}</p>
              <p className="mt-1 text-xs text-ink-2">{a.body}</p>
            </li>
          ))}
        </ol>
        <P>
          <span className="mt-4 block">
            Stack: Python (pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna), FastAPI, Next.js + TypeScript, Recharts, pytest, Vitest,
            Playwright, GitHub Actions; deployed on Vercel with a Docker image for Hugging Face Spaces. API reference:{" "}
            <a className="text-accent-ink underline" href="/api/docs">/api/docs</a>.
          </span>
        </P>

        <H>10. Data sources</H>
        <ul className="mb-4 max-w-3xl list-disc space-y-2 pl-6 leading-relaxed text-ink-2">
          <li>
            Kuzilek, J., Hlosta, M. &amp; Zdrahal, Z. (2017). Open University Learning Analytics dataset. <em>Scientific Data</em> 4, 170171.
            doi:10.1038/sdata.2017.171 — CC BY 4.0. {int(o.dataset.registrations)} registrations, {compact(o.dataset.vle_interactions)} VLE records.
          </li>
          <li>
            Realinho, V., Vieira Martins, M., Machado, J. &amp; Baptista, L. (2021). Predict Students&apos; Dropout and Academic Success. UCI Machine
            Learning Repository. doi:10.24432/C5MC89 — CC BY 4.0 (sample upload file).
          </li>
        </ul>

        {site.team.length > 0 && (
          <>
            <H>Team</H>
            <ul className="flex flex-wrap gap-3">
              {site.team.map((t) => (
                <li key={t.name} className="rounded-xl border border-line bg-surface px-4 py-3">
                  <p className="font-medium text-ink">{t.name}</p>
                  {t.role && <p className="text-sm text-ink-2">{t.role}</p>}
                </li>
              ))}
            </ul>
            {(site.institution || site.guide) && (
              <p className="mt-3 text-sm text-ink-2">{[site.institution, site.guide && `Guide: ${site.guide}`].filter(Boolean).join(" · ")}</p>
            )}
          </>
        )}
      </Container>
    </>
  );
}
