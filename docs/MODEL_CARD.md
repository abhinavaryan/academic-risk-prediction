# Model card — Agrim early-risk model

Format follows Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Numbers are from
the pipeline run of 2026-11-16; see `reports/results.md` for every table.

## Model details

| | |
|---|---|
| Deployed model | Logistic Regression (scikit-learn 1.9.1), one model per prediction day (0, 30, 60, 90, 120, 150, 180) |
| Best research model | Stacked Ensemble (not deployed: depends on libraries unsuitable for the serverless runtime) |
| Input | 49 features: demographics, course, registration timing, VLE engagement and assessment behaviour up to the prediction day |
| Output | Probability that the registration ends in Fail or Withdrawn; band (low < 33% ≤ moderate < 66% ≤ high); flag at the F1-optimal threshold; per-feature reasons |
| Storage | skops format, SHA-256 listed in `artifacts/manifest.json` |
| Version | pipeline v1.0.0 |

## Intended use

- **Primary:** early identification of students who may need academic support, so a teacher or
  counsellor can reach out.
- **Users:** teaching staff, student-support teams, learning-analytics researchers.
- **Out of scope:** admissions, grading, penalties, or any automatic decision about a student.
  The score is a prompt for a human conversation.

## Training and evaluation data

- **Data:** OULAD (Open University, UK, 2013–2014; CC BY 4.0). 32,593 registrations, 28,785 students.
- **Training data:** presentations 2013B, 2013J and 2014B (17,543 registrations active at day 60).
- **Evaluation data:** presentation 2014J (8,818 registrations), never used for training,
  model selection or threshold choice.

## Performance (day 60, 2014J test semester)

| Metric | Logistic Regression | 95% CI |
|---|---|---|
| ROC-AUC | 0.821 | 0.812–0.830 |
| F1 | 0.680 | 0.669–0.693 |
| Recall | 74.5% | 73.2%–75.8% |
| Precision | 62.6% | 61.2%–64.2% |
| Accuracy | 73.3% | |
| Brier score | 0.162 | |

Deployed model's test ROC-AUC by prediction day: day 0 → 0.722, day 30 → 0.744, day 60 → 0.821, day 90 → 0.845, day 120 → 0.879, day 150 → 0.904, day 180 → 0.925.

## Factors and fairness

The audit runs at the deployed threshold on the test semester. Largest group gaps:

| Attribute | Recall gap | False-alarm gap | Flag-rate gap |
|---|---|---|---|
| Gender | 0.050 | 0.048 | 0.005 |
| Disability | 0.054 | 0.126 | 0.151 |
| Age band | 0.300 | 0.182 | 0.260 |
| Deprivation (IMD) band | 0.307 | 0.282 | 0.375 |
| Highest education | 0.205 | 0.328 | 0.342 |

**How to read this:**
- Flag-rate differences partly mirror real differences in at-risk rates between groups.
- Recall gaps mean at-risk students in some groups are missed more often. Monitor them when
  deploying.
- Demographic factors are shown as "context" in explanations and must not be used to lower
  expectations of any student.

## Ethical considerations

- Students must not be labelled or treated differently because of a score. Use it only to offer support.
- Explanations show which observable behaviours (engagement, missed work, scores) drive a score,
  so that support targets things a student can change.
- Uploaded college data is processed in memory only. Personal identifiers are excluded from
  modelling.

## Caveats

- **Context:** one distance-learning university in 2013–14. Behaviour in other institutions and
  later years differs; retrain on local data (bring-your-own-data mode) before relying on scores.
- **Timing assumption:** assessment scores are assumed available on the submission day.
- **Not causal:** importance is predictive. Changing a feature does not guarantee a changed outcome.
