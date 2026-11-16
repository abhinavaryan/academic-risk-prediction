# Viva preparation guide

Short, honest answers to the questions examiners usually ask about an ML project. Numbers come
from `reports/results.md` (pipeline run of 2026-11-16). If you re-run the pipeline, re-check them.

---

## A. The problem and the data

**Q1. What problem does the project solve?**
It predicts, early in a course, which students are likely to **fail or withdraw**, so teachers can
help them in time. It also lets a college train the same kind of model on its own records.

**Q2. Why is this an ML problem and not a rule like "attendance < 75%"?**
Risk depends on many signals at once (engagement, assessment results, prior attempts, workload),
and their weights are not obvious. A model learns them from 32,593 historical registrations and
is checked on data it never saw. In the ablation, profile-only information reaches ROC-AUC
0.627, while all features reach 0.821.

**Q3. Which dataset, and is it real?**
OULAD, the Open University Learning Analytics Dataset (Kuzilek et al., *Scientific Data*, 2017,
CC BY 4.0): 32,593 registrations of 28,785 students, 10,655,280 VLE click records and
173,912 assessment submissions. It is anonymised, real data. We download it from the UCI
repository and verify its SHA-256 checksum.

**Q4. How did you define "at risk"?**
Final result *Fail* or *Withdrawn* = 1; *Pass* or *Distinction* = 0. 52.8% of all
registrations are at risk, so the classes are fairly balanced.

**Q5. What is a "prediction day" / cutoff?**
The day of the course on which we make the prediction. For day D we use only information that
existed on day D. We study days 0, 30, 60, 90, 120, 150, 180; day 60 is the main one.

## B. Features and leakage

**Q6. What features did you engineer?**
49 per student, in five groups:
- demographics
- course
- registration timing
- VLE engagement: total and recent clicks, active days, days since last login, weekly trend,
  clicks per activity type, engagement relative to the class median
- assessments: submitted, missed, late, average / weighted / lowest score, score relative to the
  class

**Q7. What is data leakage and how did you prevent it?**
Leakage is when the model sees information it would not have at prediction time, which makes the
results look better than reality. We:
1. build features only from events on or before the cutoff;
2. drop students who already withdrew before the cutoff;
3. exclude exams;
4. split by student, so the same person is never in both train and validation;
5. test on a later semester.

An automated test (`test_no_future_information`) adds future clicks and scores and checks that
the features do not change.

**Q8. Why group the cross-validation by student?**
3,538 students have more than one registration (for example, two modules, or a retake). If the same student sat in both the training
and validation folds, the model could memorise the person, and the scores would be inflated.

## C. Models and evaluation

**Q9. Which models did you compare?**
19 in total:
- **Single models:** Logistic Regression, Naive Bayes, k-NN, Decision Tree, Linear SVM, a neural
  network (MLP)
- **Ensembles:** Random Forest, Extra Trees, AdaBoost, HistGradientBoosting, LightGBM, XGBoost,
  CatBoost
- **Tuned versions** of four of them (Optuna)
- **A stacked ensemble**, plus a majority-class baseline

**Q10. How did you split the data?**
- **Train:** semesters 2013B, 2013J, 2014B (17,543 registrations at day 60).
- **Test:** 2014J (8,818 registrations), used once at the end.
- **Model choice and threshold:** 5-fold cross-validation inside the training data.

This is how a college would use it: learn from past years, predict the new batch.

**Q11. Why ROC-AUC as the selection metric, and why also F1?**
ROC-AUC measures how well the model *ranks* students by risk, independent of any threshold, so it
is the fairest way to compare models. F1 (with precision and recall) measures the final yes/no
decision at our chosen threshold, which is what a teacher acts on.

**Q12. What are your best results?**
At day 60 on the unseen 2014J semester:
- **Best by cross-validation:** Stacked Ensemble, test ROC-AUC 0.818.
- **Deployed:** Logistic Regression, ROC-AUC 0.821 (95% CI 0.812–0.830), F1 0.680,
  recall 74.5%, precision 62.6%.

It flags 2,505 of the 3,361 students who later failed or withdrew.

**Q13. Why is your test score lower than your cross-validation score?**
2014J is a later semester, and student behaviour and course design shift over time (distribution
shift). Cross-validation mixes the training semesters together; the test semester is genuinely
new. Reporting the lower, honest number is the point of a temporal hold-out.

**Q14. Why did you deploy Logistic Regression and not the best model?**
The live app runs on a serverless Python runtime:
- LightGBM needs a system OpenMP library that may be missing there.
- CatBoost alone is 268 MB installed.
- Scikit-learn models can be stored in the skops format and loaded without pickle, which is safer.

So we deploy the scikit-learn model with the best cross-validated ROC-AUC. The difference from the
overall best model is shown, with confidence intervals, on the Models page.

**Q15. What are the confidence intervals and McNemar's test?**
- **Bootstrap CIs:** we resample the test set 1,000 times and take the 2.5th–97.5th percentile
  of each metric. That shows how much a score could move by chance.
- **McNemar's test:** compares two models on the same students. It asks whether one model is
  right where the other is wrong more often than chance would explain. If p ≥ 0.05, the two models
  are not significantly different.

**Q16. How did you tune hyper-parameters?**
Optuna's TPE sampler (Bayesian optimisation) with 3-fold grouped cross-validation on the training
semesters. It ran 30 trials each for HistGB, LightGBM and XGBoost, and 15 for Random Forest. The
tuning-history chart shows the best score found as trials progress.

**Q17. How was the threshold chosen?**
For each model, the probability cut-off that maximises F1 on its out-of-fold training predictions.
The test set is never used for this. The deployed threshold at day 60 is 0.363.

**Q18. How early can you predict?**
Test ROC-AUC of the deployed model by prediction day: day 0 → 0.722, day 30 → 0.744, day 60 → 0.821, day 90 → 0.845, day 120 → 0.879, day 150 → 0.904, day 180 → 0.925. Accuracy grows as more
behaviour is observed, but predictions are already useful within the first weeks. By day
60, 61.9% of all withdrawals have already happened, which is why early matters.

**Q19. What does the model rely on?**
Top permutation-importance features of the deployed model: Avg. score vs. class median, Average score, Tutor-marked avg. score, Submission rate, Module. In the ablation,
engagement and assessment behaviour add the most beyond demographics.

## D. Explainability, fairness, ethics

**Q20. How do you explain an individual prediction?**
Baseline replacement: we re-score the student with one feature at a time set to the typical
student's value. The change in risk is that feature's contribution. On the Predict page each
factor is shown as "value (typical: value)" with its change in risk, in percentage points. It is exact for the deployed model and easy for
a teacher to understand. We also report SHAP values from LightGBM as a cross-check.

**Q21. Is the model fair?**
We audit recall, flag rate and false-alarm rate across gender, disability, age band, deprivation
band and prior education. The largest recall gaps are on the Models page. We also trained a
version without protected attributes (ablation) to show the trade-off. Demographic factors are
labelled "context" in the app: they explain a score but must never be used to lower expectations.

**Q22. What about privacy for a college's data?**
- Uploads are processed in memory and discarded after the response; nothing is stored or logged.
- Name, e-mail and phone columns are detected automatically and never used as features.
- API responses are not cached.

**Q23. Could this be misused?**
A risk score must start a supportive conversation (tutoring, counselling, a check-in). It should
never trigger an automatic decision such as exclusion. The model card states this.

## E. Engineering

**Q24. Explain the architecture.**
1. Data from UCI goes through the feature builder and then the research pipeline.
2. The pipeline writes three things: model files (skops, with a SHA-256 manifest), JSON for the
   dashboard, and figures plus a report.
3. A FastAPI backend serves predictions and custom training.
4. A Next.js static website is served by a CDN.
5. On Vercel, the API runs as a Python serverless function. In Docker (Hugging Face), one FastAPI
   process serves both the UI and the API.

**Q25. How did you test it?**
- **pytest:** feature-leakage tests, metrics, model registry integrity, upload parsing (encodings,
  zip bombs, oversize files), training (leak detection, determinism), the API (validation,
  security headers, rate limits, busy handling) and concurrency (50 parallel predictions, parallel
  trainings).
- **Vitest:** CSV-injection escaping and formatting.
- **Playwright:** end-to-end flows on desktop and mobile.
- **GitHub Actions:** runs everything on every push.

**Q26. What security measures did you add?**
- Model files are loaded through skops with a type allow-list and a hash check, never through
  pickle.
- Uploads are limited in size, rows and columns, and protected against zip bombs and XML attacks.
- Excel formula injection is neutralised on export.
- Request validation is strict.
- A rate limit applies, and concurrent trainings are capped.
- Security headers (CSP and others) are set, and errors never expose stack traces.

See `docs/SECURITY.md`.

**Q27. What was the hardest bug?**
CPU oversubscription. OpenMP thread limits are per-thread, and FastAPI runs requests in worker
threads. So the "one thread" limit set at start-up did not apply inside requests. Under load,
every request spawned as many threads as there are cores and they fought over the CPU: one
training took 72 s and finished only one model, versus about 17 s for all eight when limited
correctly. The fix was to apply
the limit inside each request; a regression test now checks it.

**Q28. How is it deployed for free?**
- **Main:** Vercel (static site + Python function), redeployed on every `git push`.
- **Backup:** Hugging Face Spaces (Docker), pushed by a GitHub Action.
- **Local without installs:** GitHub Codespaces.

**Q29. What would you do next?**
- Validate on a real college's anonymised records.
- Add weekly model monitoring for data drift.
- Calibrate per-college thresholds against counselling capacity.
- Explore sequence models (week-by-week engagement).
- Add a feedback loop recording which interventions helped.

**Q30. What are the limitations?**
- One UK distance-learning university, 2013–14.
- Scores are assumed known on the submission day.
- Correlation, not causation.

The custom-training mode is how the system adapts to a different college.
