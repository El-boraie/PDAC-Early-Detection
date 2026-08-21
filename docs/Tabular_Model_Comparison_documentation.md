# Clinical Model Comparison — Process, Code, and Results

**Source notebook:** `src/clinical/clinical_model_comparison.ipynb`
**Depends on:** `src/clinical/tabular_clean.ipynb` (fixed-rule cleaning → `data/processed/tabular_clean.csv`) and `src/clinical/clinical_imputer_benchmark.ipynb` (the settled `MICE_CA19_9Imputer` choice, reused unchanged here, not re-compared).
**Dataset:** Debernardi et al. 2020 urinary biomarker dataset — 590 patients, 199 PDAC / 391 not-PDAC.
**Kernel / environment:** Python 3.13.12, packages from `C:\FYP\fyp_env\Lib\site-packages`.
**Run date:** 2026-07-16. **Updated same day:** repeated CV bumped from 10 → 20 repeats (see note below); the `LogisticRegression` deprecation warning noted in Section 3 is now suppressed.

> Draft note: every number below is from the actual executed run, not estimated.

> **Update — why 20 repeats, not 10:** `n_repeats` controls how *precisely* repeated CV estimates the model's true performance, not the performance itself — averaging over more repeats narrows the noise band around the same underlying number, it cannot make a model "better." Deliberately searching for a repeat count that happens to produce a higher score would be a real methodological error (tuning the evaluation protocol to flatter a result). Going from 10 → 20 repeats was done purely because it's essentially free here (590 patients, fast models) and gives a tighter, more defensible confidence interval for the report — not because it changed any conclusion. It didn't: every repeated-CV number moved by ≤0.001, and cohort-out/leave-one-site-out are entirely unaffected by this parameter (they're single, deterministic splits, not repeated).

**Governing rules for this notebook:**
1. Three candidates only — XGBoost, regularized Logistic Regression, Random Forest. No others.
2. The CA19-9 imputer choice (`MICE_CA19_9Imputer`) is closed, not re-litigated — copied in exactly as implemented in `clinical_imputer_benchmark.ipynb`, refit fold-safe inside every fold, never on the full dataset first.
3. All three CV schemes (repeated stratified 5×20, cohort-out, leave-one-site-out) run on **identical folds/seeds across all three models** — the same splits are precomputed once and reused per model, so any performance difference is attributable to the model, not to different data landing in different folds by chance.
4. Precision, recall/sensitivity, specificity, F1, ROC-AUC, and the confusion matrix are reported for every (model × scheme) cell — never accuracy alone. Sensitivity gets its own dedicated section, not just a column in a wide table.
5. Early-stage recall also gets its own dedicated section, using a project-specific early-stage definition for this notebook (see below) — reported as directional, not as statistically stable as the other metrics.

---

## Section 1 — Load `tabular_clean.csv`

**Process:** Same column-name reconstruction pattern as `clinical_imputer_benchmark.ipynb` — `feature_matrix` (7 predictor columns), `METADATA` (identifiers + `stage`/`diagnosis`), `TARGETS` (`dx`/`target_binary`), all sharing the same index, no re-derivation of logic already verified in `tabular_clean.ipynb`.

**Result:** `feature_matrix` (590, 7), `METADATA` (590, 5), `TARGETS` (590, 2). `plasma_CA19_9` missing in 240 rows, left raw as expected.

---

## Section 2 — The Settled Imputer: `MICE_CA19_9Imputer`

**Process:** Copied verbatim from `clinical_imputer_benchmark.ipynb` — `IterativeImputer(estimator=BayesianRidge(), random_state=42)`, fit fresh on each training fold's predictors + `plasma_CA19_9`, never refit on the full dataset. Per rule 2, the imputer choice itself is closed — `KNN_CA19_9Imputer` is deliberately not copied into this notebook at all; there is no path for it to accidentally get compared again here.

---

## Section 3 — Three Candidate Models

**Process:**
- **XGBoost** — `n_estimators=100, max_depth=3` (rule 1's exact spec).
- **Logistic Regression** — regularized (`penalty='l2', C=1.0`), wrapped in `Pipeline(StandardScaler, LogisticRegression)`. `plasma_CA19_9` spans a much larger range than `age`/`sex`, so an unscaled L2 penalty would punish large-magnitude features disproportionately — a real problem for a scale-sensitive linear model, unlike XGBoost/Random Forest which split on rank order regardless of scale. The scaler is fit fresh inside every fold, internal to this one model only — the same narrow exception already established for the MICE imputer's own internal predictor scaling, never touching `feature_matrix` itself.
- **Random Forest** — `n_estimators=200, max_depth=5`, tree-based like XGBoost so needs no scaling.

Neither Logistic Regression's nor Random Forest's hyperparameters are specified by rule 1 — these are reasonable, moderate defaults, flagged here rather than silently assumed.

**Minor observed issue, now suppressed:** scikit-learn 1.8 deprecated the `penalty=` parameter on `LogisticRegression` (in favour of `l1_ratio`), so every Logistic Regression fit printed a `FutureWarning` — purely cosmetic, warnings never alter computation, confirmed by re-running: identical results with and without it. A narrowly-scoped `warnings.filterwarnings(...)` (matched on this exact message, not a blanket suppression) was added to the imports cell so it no longer clutters output, especially now that more repeats means more fits and proportionally more copies of the same warning.

---

## Section 4 — Early-Stage Definition for This Notebook

**Process:** `{I, IA, IB, II, IIA}` — a **project-specific, narrower** definition than `metrics.py`'s default `EARLY_STAGES` (which also includes `IIB`). Per the EDA's stage counts: I=1, IA=3, IB=12, II=7, IIA=11 → **34 of 590 patients (5.8%)**. The `early_stage_recall` **function** from `metrics.py` is reused completely unchanged — only the stage set passed into its existing `early_stages=` parameter differs. Nothing in `metrics.py` was modified or reimplemented.

**Result:** confirmed 34 patients under this definition (12 IB, 11 IIA, 7 II, 3 IA, 1 I).

---

## Section 5 — `run_fold`: Full Metric Suite, MICE Fixed, Model Swappable

**Process:** Fits `MICE_CA19_9Imputer` fresh on the training fold only, fits the given model, scores at the default 0.5 classification threshold (`model.predict()`), and computes the confusion matrix (`tn, fp, fn, tp`), precision, recall/sensitivity, specificity (`tn / (tn + fp)` — not directly in `sklearn.metrics`), F1, ROC-AUC, accuracy (present, never alone), and early-stage recall via the reused function.

**A real correctness fix made during this build, not scope creep:** the first execution used `zero_division=0` uncritically for every metric, which silently reported `0.0` for recall/F1 whenever a test fold had zero actual positive cases — this happens for the `UCL` site in leave-one-site-out, which is 100% Benign (0 PDAC patients). Recall is mathematically **undefined** there (0/0), not 0 — the same "report `NaN`, not a manufactured number" principle already used everywhere else in this project (AUC, PR-AUC, early-stage recall all already do this for single-class folds). Fixed: `recall`/`f1` are now `NaN` when the test fold has zero true positives available to detect; `precision` is only `NaN` when the model also predicts zero positives (a rarer, separate edge case); `specificity` is unaffected (well-defined regardless of whether any positives exist in the test fold). This changed the leave-one-site-out recall means substantially — see Section 8.

When `oof_rows` is passed (repeated-CV scheme only), every test-fold prediction is appended for future calibration use.

---

## Section 6 — Repeated Stratified 5-Fold × 20-Repeat CV

**Process:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)`, 100 folds precomputed once and shared across all three models. Main performance estimate — the only scheme with enough repeats to average out fold-to-fold noise. (Originally run at 10 repeats; bumped to 20 for a tighter estimate — see the "Update" note at the top of this document. The first 10 repeats are deterministically identical to the original run, since `RepeatedStratifiedKFold` derives each repeat's seed in sequence from the same base `random_state`; repeats 11–20 are new.)

**Result:**

| Model | AUC | Precision | Recall (Sensitivity) | Specificity | F1 | Early Recall (n=680*) |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.9077 ± 0.0224 | 0.7760 ± 0.0578 | **0.7433 ± 0.0554** | 0.8884 ± 0.0364 | **0.7573** | **0.6379 ± 0.1965** |
| Logistic Regression | 0.8786 ± 0.0304 | 0.7877 ± 0.0641 | 0.6157 ± 0.0609 | 0.9130 ± 0.0330 | 0.6884 | 0.5168 ± 0.2089 |
| Random Forest | 0.9068 ± 0.0239 | 0.7943 ± 0.0662 | 0.6920 ± 0.0632 | 0.9060 ± 0.0361 | 0.7370 | 0.6109 ± 0.1930 |

*n=680 is 34 unique early-stage patients × 20 CV repeats, not 680 independent patients — see Section 9. Every number above moved by ≤0.001 from the original 10-repeat run — a tighter estimate of the same quantity, not a different one (see the "Update" note at the top).

Confusion matrix totals (summed across all 100 folds, 11,800 total predictions = 590 patients × 20 repeats):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| XGBoost | 6947 | 873 | 1022 | 2958 |
| Logistic Regression | 7140 | 680 | 1530 | 2450 |
| Random Forest | 7085 | 735 | 1226 | 2754 |

XGBoost and Random Forest are essentially tied on AUC (0.9077 vs. 0.9068, a 0.0009 gap — negligible); XGBoost leads clearly on recall/sensitivity (+0.05 over Random Forest, +0.13 over Logistic Regression), F1, and early-stage recall, while trailing marginally on specificity/precision (~0.01–0.02, much smaller than its recall advantage).

---

## Section 7 — Cohort-Out Split (train = Cohort1, test = Cohort2)

**Process:** Train on `patient_cohort == 'Cohort1'` (n=332), test on `'Cohort2'` (n=258) — a **single** split, not an average over repeats, so the noisiest of the three estimates here.

**Result:**

| Model | AUC | Precision | Recall (Sensitivity) | Specificity | F1 | Early Recall (n=10) |
|---|---|---|---|---|---|---|
| XGBoost | 0.8375 | 0.5319 | 0.6757 | 0.9005 | 0.5952 | 0.5000 |
| Logistic Regression | 0.8793 | 0.5306 | 0.7027 | 0.8959 | 0.6047 | 0.5000 |
| **Random Forest** | **0.8901** | **0.5600** | **0.7568** | 0.9005 | **0.6437** | **0.6000** |

Confusion matrix (single split):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| XGBoost | 199 | 22 | 12 | 25 |
| Logistic Regression | 198 | 23 | 11 | 26 |
| Random Forest | 199 | 22 | 9 | 28 |

Random Forest wins clearly on every metric in this scheme; XGBoost is weakest here specifically.

---

## Section 8 — Leave-One-Site-Out (over `sample_origin`)

**Process:** For each of BPTB, ESP, LIV, UCL: train on the other three sites, test on the held-out one — the strictest generalization test. `UCL` (n=20) is 100% Benign — 0 PDAC patients — so AUC, recall, F1, and early-stage recall are all mathematically undefined there and correctly reported as `NaN` (see Section 5's fix), not averaged in.

**Per-site detail:**

| Model | Site | n test | AUC | Recall | Specificity | F1 | Early Recall | n early |
|---|---|---|---|---|---|---|---|---|
| XGBoost | BPTB | 409 | 0.842 | 0.723 | 0.782 | 0.561 | 0.667 | 9 |
| XGBoost | ESP | 29 | 0.783 | 0.913 | 0.333 | 0.875 | 1.000 | 8 |
| XGBoost | LIV | 132 | 0.834 | 0.699 | 0.795 | 0.783 | 0.588 | 17 |
| XGBoost | UCL | 20 | NaN | NaN | 0.850 | NaN | NaN | 0 |
| Logistic Regression | BPTB | 409 | 0.819 | 0.783 | 0.650 | 0.496 | 0.778 | 9 |
| Logistic Regression | ESP | 29 | 0.826 | 0.957 | 0.500 | 0.917 | 1.000 | 8 |
| Logistic Regression | LIV | 132 | 0.821 | 0.505 | 0.846 | 0.644 | 0.294 | 17 |
| Logistic Regression | UCL | 20 | NaN | NaN | 0.800 | NaN | NaN | 0 |
| Random Forest | BPTB | 409 | 0.845 | 0.747 | 0.758 | 0.554 | 0.556 | 9 |
| Random Forest | ESP | 29 | 0.841 | 0.913 | 0.667 | 0.913 | 1.000 | 8 |
| Random Forest | LIV | 132 | 0.811 | 0.527 | 0.821 | 0.658 | 0.412 | 17 |
| Random Forest | UCL | 20 | NaN | NaN | 0.750 | NaN | NaN | 0 |

**Aggregate (mean across BPTB/ESP/LIV, UCL correctly excluded):**

| Model | AUC | Precision | Recall (Sensitivity) | Specificity | F1 | Early Recall (n=34) |
|---|---|---|---|---|---|---|
| XGBoost | 0.8195 ± 0.0322 | 0.5471 ± 0.4127 | **0.7783 ± 0.1173** | 0.6901 ± 0.2397 | **0.7396** | **0.7516 ± 0.2186** |
| Logistic Regression | 0.8221 ± 0.0036 | 0.5325 ± 0.4315 | 0.7483 ± 0.2276 | 0.6991 ± 0.1569 | 0.6856 | 0.6906 ± 0.3609 |
| **Random Forest** | **0.8324 ± 0.0183** | 0.5569 ± 0.4289 | 0.7290 ± 0.1937 | **0.7487 ± 0.0632** | 0.7081 | 0.6558 ± 0.3067 |

Confusion matrix totals (summed across the 4 sites, including UCL's real TN/FP counts):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| XGBoost | 305 | 86 | 53 | 146 |
| Logistic Regression | 264 | 127 | 65 | 134 |
| Random Forest | 298 | 93 | 67 | 132 |

Precision's very large standard deviation here (0.41+) is real, not a bug — it's driven by genuine site-to-site variation across only 4 sites (one of which, UCL, has a structurally different, degenerate case). Random Forest has the best AUC and specificity; XGBoost has the best recall, F1, and early-stage recall.

---

## Section 9 — Sensitivity (Recall) — Dedicated Section

Per rule 4, pulled out on its own — sensitivity is the single most clinically important number here, since missing a PDAC case is far costlier than a false alarm.

| Scheme | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Repeated 5×20 CV | 0.6157 | 0.6920 | **0.7433** |
| Cohort-out | 0.7027 | **0.7568** | 0.6757 |
| Leave-one-site-out | 0.7483 | 0.7290 | **0.7783** |

XGBoost has the best sensitivity in 2 of 3 schemes — the main repeated-CV estimate and the hardest generalization test (leave-one-site-out) — and only trails Random Forest in the single, non-repeated cohort-out split.

---

## Section 10 — Early-Stage Recall — Dedicated Section (Directional Only)

**Read this as directional context, not a stable estimate.** Only 34 of 590 patients (~5.8%) fall under this notebook's early-stage definition (`I`/`IA`/`IB`/`II`/`IIA`). The "n" column below is patient-**instances** scored, not unique patients — repeated 5×20 CV scores the same 34 patients across 20 repeats (34 × 20 = 680 instances, not 680 different people); leave-one-site-out and cohort-out score the true 34 once each. A handful of patients flipping from correctly- to incorrectly-classified swings this number by double-digit percentage points — treat differences between models here as suggestive, not decisive, unlike the other metrics above.

| Scheme | Logistic Regression | Random Forest | XGBoost | Patients behind this |
|---|---|---|---|---|
| Repeated 5×20 CV | 0.5168 | 0.6109 | **0.6379** | 34 unique (680 instances) |
| Cohort-out | 0.5000 | **0.6000** | 0.5000 | 10 |
| Leave-one-site-out | 0.6906 | 0.6558 | **0.7516** | 34 |

XGBoost leads in 2 of 3 schemes here too, consistent with its sensitivity lead above — but given the small-sample caveat, this reinforces rather than independently establishes the recommendation below.

---

## Section 11 — Outputs

- **`results/clinical/model_comparison.csv`** — 9 rows (3 models × 3 schemes), all metrics above plus summed confusion-matrix counts (`tn`/`fp`/`fn`/`tp`). **Overwritten in full on every run** — this file (and `oof_predictions.csv`) hold only the most recent run's results, not a running history across parameter changes.
- **`results/clinical/oof_predictions.csv`** — 35,400 rows (590 patients × 20 CV repeats × 3 models), collected during the repeated-CV scheme only, for future calibration use (`clinical_final_fit.ipynb`, not yet rebuilt after the earlier rollback).

## Section 12 — Consistency Check Against `clinical_imputer_benchmark.ipynb`

**Not a hard gate** — nothing in this notebook depends on it passing — just a transparency check that XGBoost + MICE here (identical imputer, hyperparameters, data, and fold seeds) reproduces the AUC numbers already established when MICE was validated. **Cohort-out and leave-one-site-out are single, deterministic splits, entirely unaffected by `n_repeats`, so they stay hard-checked against the original verified numbers.** Repeated CV is no longer checked against a fixed target — 20 repeats is a different (tighter) estimate of the same quantity than the original 10-repeat run by design, so it's reported for information only, not as a pass/fail.

| Scheme | Expected AUC | Actual AUC | Result |
|---|---|---|---|
| Cohort-out | 0.8375 | 0.8375 | MATCH |
| Leave-one-site-out | 0.8195 | 0.8195 | MATCH |
| Repeated 5×20 CV | — (informational) | 0.9077 | — |

---

## Final Summary

### Verdict: **XGBoost** — recommended winner.

**Reasoning, not just the numbers:** XGBoost has the best sensitivity/recall — the metric that matters most here, since missing a PDAC case is far costlier than a false alarm — in **2 of the 3 schemes**: the main repeated 5×20 CV estimate (0.7433, vs. Random Forest's 0.6920 and Logistic Regression's 0.6157) and leave-one-site-out, the strictest generalization test (0.7783, vs. 0.7290 and 0.7483). It also leads early-stage recall in those same two schemes, though that metric is explicitly directional given only 34 patients.

On AUC specifically, XGBoost and Random Forest are close enough to call negligible in the two schemes that matter most (repeated CV: 0.9077 vs. 0.9068, a 0.0009 gap; leave-one-site-out: 0.8195 vs. 0.8324, a 0.013 gap) — not a case for picking Random Forest on AUC alone when XGBoost's recall advantage in the same schemes is 4–5x larger in absolute terms.

**Random Forest only wins clearly in cohort-out** — every metric there favours it over XGBoost. But cohort-out is a **single train/test split** (Cohort1 → Cohort2), not an average over repeats — the statistically weakest of the three estimates, exactly as this branch has already established for the KNN-vs-MICE comparison. One volatile single-split result isn't enough to override two multi-fold schemes that both favour XGBoost on the metric prioritised here.

This also happens to validate rather than contradict the architecturally "locked" choice of XGBoost from `PROJECT_HANDOFF.md` — this comparison isn't rubber-stamping that decision; sensitivity, not AUC, is what actually settles it, and the two are far from perfectly correlated here (Random Forest's AUC edge in leave-one-site-out comes with a real recall cost).

**Not negligible, but not overwhelming either** — stated plainly: XGBoost's recall lead over Random Forest is real and consistent (5-6 percentage points in both of the two more reliable schemes) but this is 590 patients, three CV schemes, and hyperparameters that weren't independently tuned per model. If either model's hyperparameters were tuned harder, this gap could narrow or shift. Worth re-deriving if the predictor set or model hyperparameters change materially.
