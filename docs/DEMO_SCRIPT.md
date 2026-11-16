# Demo script (6–8 minutes)

Open the live site before you start (a cold serverless function or sleeping Space takes a few
seconds to wake up). Keep the Hugging Face link open in a second tab as a backup.

## 1. The problem (Overview page, 1 min)

- "Universities find out a student is struggling when the exam result arrives — too late. Our
  system flags at-risk students in the first weeks of a course."
- Point at the hero number: the share of students who later failed or withdrew that the model
  flagged at day 60, measured on a semester the model never saw.
- Point at the withdrawal curve: most withdrawals happen early, so early prediction matters.

## 2. Real data (Dataset page, 1 min)

- 32,593 real registrations, 10.6 million click records from the Open University (OULAD).
- The engagement chart: students who later fail or withdraw are less active from the first weeks.
  "This is the signal the models learn."

## 3. Many models, compared honestly (Models page, 2 min)

- Leaderboard: every model, cross-validation score and test score with confidence intervals.
- Explain the protocol in one sentence: "trained on 2013–2014B, selected by grouped
  cross-validation, then tested once on 2014J — a later semester — like a real deployment."
- Early-warning timeline: accuracy improves as the course goes on; we can still predict usefully at day 30.
- Ablation: demographics alone are weak; engagement and assessments add the most.
- Fairness audit: recall per gender / disability / age / deprivation.

## 4. Live prediction with explanations (Predict page, 1–2 min)

- Load a random real student from the test semester, show the risk and the factors.
- Drag "Total LMS clicks" or "Assessments missed" and watch the risk move (what-if analysis).
- Click "Reveal actual outcome".

## 5. Train on your own college's data (Train page, 1–2 min)

- Click **Try the sample dataset** (4,424 real university students from Portugal).
- Show the auto-detected result column and value mapping (Dropout = at risk, Graduate = not,
  Enrolled = unknown → scored).
- Train: 8 models are compared in seconds; enrolled students get a risk score with reasons.
- Download the CSV report.
- "Ma'am, if you give us an anonymised sheet from our college — attendance, internal marks,
  assignments, previous result — it trains on that in the same way. Nothing is stored."

## 6. Engineering (30 s, if asked)

API docs at `/api/docs`, automated tests (Python, web, end-to-end), CI on every push, security
controls (see `docs/SECURITY.md`), reproducible pipeline (`make research`).
