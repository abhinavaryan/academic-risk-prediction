# Results — early academic-risk prediction on OULAD

_Generated 2026-11-16T20:37:46+00:00 by `python -m riskml.research.pipeline` (pipeline v1.0.0, runtime 33.2 min). Every number below is computed from the public dataset._

## Dataset

| Item | Value |
|---|---|
| Course registrations | 32,593 |
| Unique students | 28,785 |
| Modules / course runs | 7 / 22 |
| VLE activity records | 10,655,280 (39,605,099 clicks) |
| Assessment submissions | 173,912 |

## Protocol

- Train: 2013B, 2013J, 2014B (17,543 registrations at day 60); test: 2014J (8,818).

- Model selection: 5-fold StratifiedGroupKFold grouped by student, metric = CV ROC-AUC. Decision threshold maximises F1 on out-of-fold predictions.

- 95% confidence intervals: percentile bootstrap (1,000 resamples) of the test set.

## Model leaderboard (day 60)

| # | Model | CV ROC-AUC | Test ROC-AUC [95% CI] | F1 | Precision | Recall | Accuracy | MCC | McNemar p vs best |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Stacked Ensemble | 0.849 ± 0.005 | 0.818 [0.809–0.827] | 0.684 | 60.6% | 78.5% | 72.3% | 0.456 | — |
| 2 | XGBoost (tuned) | 0.846 ± 0.005 | 0.805 [0.796–0.815] | 0.681 | 61.7% | 76.1% | 72.9% | 0.458 | 0.006 |
| 3 | LightGBM (tuned) | 0.846 ± 0.004 | 0.804 [0.794–0.813] | 0.678 | 62.1% | 74.5% | 73.0% | 0.454 | 0.005 |
| 4 | Logistic Regression **(deployed)** | 0.846 ± 0.005 | 0.821 [0.812–0.830] | 0.680 | 62.6% | 74.5% | 73.3% | 0.460 | <0.001 |
| 5 | CatBoost | 0.845 ± 0.005 | 0.813 [0.804–0.822] | 0.676 | 62.6% | 73.6% | 73.2% | 0.454 | 0.001 |
| 6 | Gradient Boosting (HistGB) (tuned) | 0.845 ± 0.004 | 0.803 [0.793–0.812] | 0.677 | 61.2% | 75.8% | 72.5% | 0.449 | 0.522 |
| 7 | Neural Network (MLP) | 0.844 ± 0.007 | 0.811 [0.802–0.821] | 0.675 | 62.6% | 73.2% | 73.1% | 0.452 | 0.006 |
| 8 | Gradient Boosting (HistGB) | 0.842 ± 0.003 | 0.800 [0.790–0.809] | 0.676 | 61.9% | 74.4% | 72.8% | 0.450 | 0.067 |
| 9 | LightGBM | 0.842 ± 0.005 | 0.803 [0.794–0.813] | 0.677 | 60.0% | 77.6% | 71.8% | 0.445 | 0.042 |
| 10 | Random Forest (tuned) | 0.842 ± 0.004 | 0.804 [0.794–0.813] | 0.675 | 60.6% | 76.3% | 72.0% | 0.444 | 0.288 |
| 11 | Random Forest | 0.841 ± 0.004 | 0.805 [0.795–0.814] | 0.676 | 61.0% | 75.8% | 72.3% | 0.446 | 0.900 |
| 12 | XGBoost | 0.841 ± 0.004 | 0.806 [0.796–0.815] | 0.675 | 62.0% | 74.0% | 72.8% | 0.449 | 0.102 |
| 13 | Extra Trees | 0.840 ± 0.004 | 0.797 [0.787–0.806] | 0.671 | 59.6% | 76.8% | 71.3% | 0.435 | <0.001 |
| 14 | Linear SVM | 0.835 ± 0.006 | 0.801 [0.792–0.811] | 0.664 | 60.4% | 73.8% | 71.6% | 0.429 | 0.033 |
| 15 | AdaBoost | 0.829 ± 0.004 | 0.791 [0.782–0.800] | 0.663 | 58.7% | 76.1% | 70.5% | 0.419 | <0.001 |
| 16 | k-Nearest Neighbours | 0.825 ± 0.006 | 0.776 [0.766–0.786] | 0.652 | 61.0% | 69.9% | 71.5% | 0.415 | 0.037 |
| 17 | Decision Tree | 0.812 ± 0.004 | 0.769 [0.758–0.779] | 0.657 | 56.4% | 78.7% | 68.7% | 0.401 | <0.001 |
| 18 | Naive Bayes | 0.746 ± 0.003 | 0.700 [0.690–0.710] | 0.552 | 38.1% | 100.0% | 38.1% | 0.000 | <0.001 |
| 19 | Baseline (majority) | 0.500 ± 0.000 | 0.500 [0.500–0.500] | 0.552 | 38.1% | 100.0% | 38.1% | 0.000 | <0.001 |

Best by cross-validation: **Stacked Ensemble**. Deployed: **Logistic Regression** — Highest cross-validated ROC-AUC among scikit-learn-native models, which run on the serverless runtime without extra native libraries and load securely with skops.

## Early-warning timeline (test ROC-AUC / F1 by prediction day)

| Day | Students still enrolled | At-risk share | Logistic Regression AUC / F1 | Random Forest AUC / F1 | Gradient Boosting (HistGB) AUC / F1 | LightGBM AUC / F1 |
|---|---|---|---|---|---|---|
| 0 | 10,363 | 47.3% | 0.722 / 0.677 | 0.714 / 0.676 | 0.728 / 0.682 | 0.724 / 0.680 |
| 30 | 9,203 | 40.7% | 0.744 / 0.659 | 0.749 / 0.651 | 0.754 / 0.660 | 0.755 / 0.659 |
| 60 | 8,818 | 38.1% | 0.821 / 0.680 | 0.804 / 0.675 | 0.803 / 0.677 | 0.804 / 0.678 |
| 90 | 8,533 | 36.0% | 0.845 / 0.695 | 0.828 / 0.672 | 0.827 / 0.676 | 0.831 / 0.679 |
| 120 | 8,260 | 33.9% | 0.879 / 0.732 | 0.867 / 0.711 | 0.876 / 0.723 | 0.878 / 0.724 |
| 150 | 8,010 | 31.9% | 0.904 / 0.755 | 0.899 / 0.741 | 0.903 / 0.755 | 0.903 / 0.752 |
| 180 | 7,719 | 29.3% | 0.925 / 0.782 | 0.921 / 0.764 | 0.924 / 0.766 | 0.924 / 0.766 |

## Feature-group ablation (Logistic Regression)

| Feature set | Features | ROC-AUC | F1 | Recall |
|---|---|---|---|---|
| Demographics only | 8 | 0.627 | 0.560 | 85.9% |
| + Course & registration | 12 | 0.648 | 0.573 | 81.7% |
| + Assessments | 26 | 0.800 | 0.661 | 75.2% |
| All features (+ VLE engagement) | 49 | 0.821 | 0.681 | 75.3% |
| All except protected attributes | 44 | 0.819 | 0.679 | 73.9% |

## Most important features (permutation importance, deployed model)

| Feature | Drop in test ROC-AUC |
|---|---|
| Avg. score vs. class median | 0.1466 ± 0.0037 |
| Average score | 0.0772 ± 0.0033 |
| Tutor-marked avg. score | 0.0409 ± 0.0042 |
| Submission rate | 0.0372 ± 0.0025 |
| Module | 0.0353 ± 0.0015 |
| Weighted average score | 0.0253 ± 0.0012 |
| Active days on LMS | 0.0236 ± 0.0034 |
| Assessments submitted | 0.0206 ± 0.0022 |
| Days since last LMS activity | 0.0161 ± 0.0018 |
| Avg. days submitted before deadline | 0.0078 ± 0.0021 |
| Late submissions | 0.0078 ± 0.0015 |
| Lowest score | 0.0070 ± 0.0010 |
| Highest education | 0.0063 ± 0.0008 |
| Computer-marked avg. score | 0.0050 ± 0.0011 |
| Banked (carried-over) results | 0.0037 ± 0.0006 |

## Fairness audit (deployed model, test semester)

| Attribute | Recall gap | False-alarm gap | Flag-rate gap | Recall gap without protected attributes |
|---|---|---|---|---|
| Gender | 0.050 | 0.048 | 0.005 | 0.053 |
| Disability | 0.054 | 0.126 | 0.151 | 0.029 |
| Age band | 0.300 | 0.182 | 0.260 | 0.125 |
| Deprivation (IMD) band | 0.307 | 0.282 | 0.375 | 0.141 |
| Highest education | 0.205 | 0.328 | 0.342 | 0.271 |

## Figures

See `reports/figures/` (PNG, 200 dpi).
