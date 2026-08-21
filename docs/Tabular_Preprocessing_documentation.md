# Tabular Preprocessing Pipeline — Process, Code, and Results

**Source notebook:** `src/clinical/clinical.ipynb`
**Dataset:** Debernardi et al. 2020 urinary biomarker dataset — 590 patients, 3 diagnosis classes (Control/Benign/PDAC)
**Depends on:** `notebooks/Tabular EDA.ipynb` / `docs/Tabular_EDA_documentation.md` — every decision below traces back to an EDA finding, not an arbitrary default.
**Target model:** XGBoost (tree-based, scale/monotonic-transform invariant — this is why the no-transform rule below is safe, not just permitted)
**Kernel / environment:** Python 3.13.12, packages from `C:\FYP\fyp_env\Lib\site-packages`
**Run date:** 2026-07-14, initial single-imputer (KNN) build. **Updated 2026-07-14:** extended to a two-imputer (KNN vs. MICE) empirical comparison — see Sections 7-9 below. Full pipeline (load through all six comparison runs) executes in well under a minute.

> Draft note: every number below is from the actual executed run, not estimated.

**Governing rules for this whole notebook, carried over unchanged from the build spec:**
1. `plasma_CA19_9` is a feature, always — third-highest Pearson correlation with diagnosis (r=0.26) and highest Kruskal-Wallis H-statistic of all six biomarkers (209.01, p=4.1e-46) per the EDA.
2. No log transform, no winsorizing, no outlier removal, no scaling of the **final feature matrix**, anywhere. These are lab test results — skew and outliers are real disease signal, and XGBoost splits on rank order, so none of this helps and all of it would destroy information. The one narrow exception: the KNN imputer's *internal* predictor scaling (Section 7) — that `StandardScaler` only exists to make Euclidean distance meaningful for the nearest-neighbour search: it never touches the values that actually reach XGBoost.
3. `plasma_CA19_9` is never zero-filled and never gets a missingness indicator, under either imputer — 0 is a real, physiologically valid concentration for this biomarker, so zero-filling would fabricate a clinical measurement.
4. Both `plasma_CA19_9` imputers are refit inside every single CV fold, on that fold's training rows only, never on the full dataset first — the same principle as fitting a `StandardScaler` only on training data.
5. Neither imputer predicts `plasma_CA19_9` from anything but `creatinine`, `LYVE1`, `REG1B`, `TFF1`, `age` — never from `diagnosis`, `dx`, or `target_binary`, since that would make imputation a label proxy and leak the target into the inputs regardless of how correctly the CV splits are done.
6. No imputer is picked by assumption or convenience — the winner is decided only by running the full downstream pipeline with each and comparing results (Section 9).

---

## Section 1 — Load Raw Data

**Process:** `pd.read_csv()` on the raw file, no cleaning, no dtype coercion, no transforms.

**Result:** shape `(590, 14)`, columns: `sample_id, patient_cohort, sample_origin, age, sex, diagnosis, stage, benign_sample_diagnosis, plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A`.

---

## Section 2 — Derive Target Labels

**Process:** `diagnosis` (1=Control, 2=Benign, 3=PDAC) is mapped to a readable `dx` label and a binary `target_binary` (1 = PDAC, else 0). Both are targets only — never touched by the imputer (rule 5) and never placed in `FEATURES`.

```python
DX_MAP = {1: "Control", 2: "Benign", 3: "PDAC"}
df["dx"] = df["diagnosis"].map(DX_MAP)
df["target_binary"] = (df["diagnosis"] == 3).astype(int)
```

**Result:**

| dx | count |
|---|---|
| Benign | 208 |
| PDAC | 199 |
| Control | 183 |

`target_binary`: 391 not-PDAC, 199 PDAC.

---

## Section 3 — Sentinel-Fill `stage` and `benign_sample_diagnosis`

**Process:** Both columns are missing by design, not by accident — `stage` only applies to PDAC patients, `benign_sample_diagnosis` only to Benign patients. Rather than trust that exclusivity as given, it's re-verified programmatically after filling with explicit sentinel categories (`'No Cancer'`, `'Control/PDAC'`) rather than any statistically-estimated value — "not applicable" isn't a value a model can predict.

```python
metadata_sentinel["stage"] = df["stage"].fillna("No Cancer")
metadata_sentinel["benign_sample_diagnosis"] = df["benign_sample_diagnosis"].fillna("Control/PDAC")

check_stage = (metadata_sentinel["stage"] != "No Cancer") == (df["diagnosis"] == 3)
check_benign = (metadata_sentinel["benign_sample_diagnosis"] != "Control/PDAC") == (df["diagnosis"] == 2)
assert check_stage.all() and check_benign.all()
```

**Result:** both equivalences hold **exactly**, on all 590 rows:
- `(stage != 'No Cancer') == (diagnosis == 3)` → `True`
- `(benign_sample_diagnosis != 'Control/PDAC') == (diagnosis == 2)` → `True`

Neither filled column enters `FEATURES`; `stage` (sentinel-filled) carries forward into `METADATA` in Section 6, `benign_sample_diagnosis` does not carry forward past this verification step.

---

## Section 4 — Drop `REG1A`

**Process:** The EDA's sharpest missingness finding: `REG1A` is missing in 258/258 (100%) of Cohort2 patients versus 7.8% of Cohort1 — not spread-out missingness, an entire cohort never ran the assay. Any imputation for it would invent values for a whole cohort with zero ground truth to anchor against, and a missingness indicator would just re-encode `patient_cohort`, which already exists. Dropped outright — no impute, no indicator.

```python
df = df.drop(columns=["REG1A"])
```

**Result:** shape `(590, 15)` (14 raw columns − `REG1A` + `dx` + `target_binary` from Section 2).

---

## Section 5 — Build `FEATURES`

**Process:** `FEATURES = ['creatinine', 'LYVE1', 'REG1B', 'TFF1', 'plasma_CA19_9', 'age', 'sex']`. `sex` is encoded 0/1 (`F=0`, `M=1` — arbitrary but fixed) since XGBoost needs numeric input; no other column is encoded, scaled, or transformed. `plasma_CA19_9`'s 240 missing values are left as `NaN` here on purpose — filling them at this stage, before any train/test split exists, is exactly the leak rule 4 exists to prevent.

**Result:** `feature_matrix` shape `(590, 7)`.

| feature | missing |
|---|---|
| creatinine | 0 |
| LYVE1 | 0 |
| REG1B | 0 |
| TFF1 | 0 |
| plasma_CA19_9 | 240 |
| age | 0 |
| sex | 0 |

---

## Section 6 — Build `METADATA`

**Process:** `METADATA = ['sample_id', 'patient_cohort', 'sample_origin', 'stage', 'diagnosis']`, aligned to `feature_matrix` by index, never merged into it. `dx`/`target_binary` live in their own `TARGETS` frame instead of `METADATA` — keeping targets structurally separate from both features and metadata makes it impossible to accidentally concatenate them into a feature matrix later.

**Judgment call, flagged for confirmation:** `METADATA`'s `stage` column uses the **sentinel-filled** version from Section 3 (`'No Cancer'` instead of `NaN`), not the raw column. This wasn't explicitly specified in the build requirements — it was chosen because `stage` here is bookkeeping, not a modeling input, so a readable category is strictly more useful than a null for downstream grouping/plotting, with no leakage risk since it never reaches the model. Easy to switch to the raw column if preferred.

**Result:** `feature_matrix` `(590, 7)`, `METADATA` `(590, 5)`, `TARGETS` `(590, 2)` — all three share the same index.

---

## Section 7 — Two `plasma_CA19_9` Imputers, Built to Compare

**Process:** the original build picked `KNNImputer` by judgment call (see the now-resolved note in "Open Judgment Calls" below). This version doesn't pick — it builds **two** candidate imputers with an identical `fit(train_df)`/`transform(target_df)` interface (mirroring a scikit-learn transformer) and lets Section 9's actual downstream performance decide which one is used going forward. Both share the same fold-safety rules: fit only on the training fold, predict `plasma_CA19_9` only from the five approved predictors, never zero-fill, never add a missingness indicator.

**`KNN_CA19_9Imputer`** (the original class, renamed, logic unchanged):
1. `fit(train_df)` — takes only the training fold's rows where `plasma_CA19_9` is observed, fits a `StandardScaler` on their five predictors, then fits `KNNImputer(n_neighbors=5)` on the scaled predictors + observed `plasma_CA19_9`.
2. `transform(target_df)` — scales with the **already-fitted** scaler (never refit), then the fitted `KNNImputer` fills missing values via the 5-nearest-neighbour average, drawn only from the training fold's observed-`CA19_9` rows.

```python
class KNN_CA19_9Imputer:
    PREDICTORS = ["creatinine", "LYVE1", "REG1B", "TFF1", "age"]
    TARGET = "plasma_CA19_9"

    def __init__(self, n_neighbors=5):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, train_df):
        observed = train_df[train_df[self.TARGET].notna()]
        self.scaler.fit(observed[self.PREDICTORS])
        scaled = self.scaler.transform(observed[self.PREDICTORS])
        self.imputer.fit(np.column_stack([scaled, observed[self.TARGET].values]))
        return self

    def transform(self, target_df):
        out = target_df.copy()
        scaled = self.scaler.transform(out[self.PREDICTORS])
        matrix = np.column_stack([scaled, out[self.TARGET].values])
        out[self.TARGET] = self.imputer.transform(matrix)[:, -1]
        return out
```

**`MICE_CA19_9Imputer`** (new): fits `sklearn.impute.IterativeImputer(estimator=BayesianRidge(), random_state=42)` directly on the training fold's predictors (always complete) + `plasma_CA19_9` (partially missing) — no separate scaler, since that's exactly how `IterativeImputer` is meant to be used. It fits a `BayesianRidge` regression of the target on the predictors using the observed rows, then uses that model to fill the missing ones, refined over a few iterations. `transform()` reuses the already-fitted imputer, same never-refit-on-target-data rule as KNN.

```python
class MICE_CA19_9Imputer:
    PREDICTORS = ["creatinine", "LYVE1", "REG1B", "TFF1", "age"]
    TARGET = "plasma_CA19_9"

    def __init__(self, random_state=42):
        self.imputer = IterativeImputer(estimator=BayesianRidge(), random_state=random_state)

    def fit(self, train_df):
        self.imputer.fit(train_df[self.PREDICTORS + [self.TARGET]])
        return self

    def transform(self, target_df):
        out = target_df.copy()
        out[self.TARGET] = self.imputer.transform(out[self.PREDICTORS + [self.TARGET]])[:, -1]
        return out
```

Neither class ever reads `diagnosis`, `dx`, or `target_binary` — both hardcode the same five-column predictor list.

**Smoke test (single random 80/20 split, not yet a CV fold, same split reused for both imputers):**

| imputer | split | rows | missing before | missing after |
|---|---|---|---|---|
| KNN | train (80%) | 473 | 199 | 0 |
| KNN | held-out (20%) | 117 | 41 | 0 |
| MICE | train (80%) | 473 | 199 | 0 |
| MICE | held-out (20%) | 117 | 41 | 0 |

Both classes fit cleanly on the training subset and fill both themselves and the untouched held-out set correctly before being wired into real CV.

---

## Section 8 — A Shared, Imputer-Agnostic `run_fold`

**Process:** `run_fold` now takes an `imputer_class` argument instead of being hardwired to one imputer — the same fold-safety logic (fit fresh on the training rows only, transform both sides, fit XGBoost, score) runs identically regardless of which imputer is plugged in, so Section 9's comparison isn't accidentally comparing "KNN done correctly" against "MICE done slightly differently." Both `KNN_CA19_9Imputer()` and `MICE_CA19_9Imputer()` can be constructed with no arguments (their tunable defaults are baked in), so `run_fold` just calls `imputer_class()`.

It also now returns **four** metrics instead of two:
- **AUC** and **accuracy**, as before.
- **PR-AUC** (`average_precision_score`) — more informative than ROC-AUC under class imbalance (391 not-PDAC vs. 199 PDAC), since it's sensitive to precision at the positive class specifically.
- **Early-stage recall** — of the test fold's PDAC patients whose `stage` is early (`I`/`IA`/`IB`/`II`/`IIA`/`IIB` — a resectability-based cut, **judgment call, flagged for confirmation**, not a canonical clinical threshold from the EDA), what fraction did the model correctly flag as PDAC? This is the metric that matters most for a screening tool — catching late-stage PDAC is far less clinically useful than catching it early, so overall recall could look fine while quietly missing most early cases. Computable directly from `METADATA['stage']`, already in scope (non-null exclusively for PDAC patients per Section 3's verified sentinel-fill) — no separate evaluation script needed.

Both new metrics get `NaN` under the same conditions AUC already does — reported plainly, never raising or silently substituting a default.

```python
EARLY_STAGES = {"I", "IA", "IB", "II", "IIA", "IIB"}

def run_fold(train_idx, test_idx, imputer_class, feature_matrix, targets, metadata, random_state=0):
    train_fold, test_fold = feature_matrix.loc[train_idx], feature_matrix.loc[test_idx]
    y_train, y_test = targets.loc[train_idx, "target_binary"], targets.loc[test_idx, "target_binary"]

    imputer = imputer_class().fit(train_fold)
    train_imp, test_imp = imputer.transform(train_fold), imputer.transform(test_fold)

    model = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=random_state)
    model.fit(train_imp[FEATURES], y_train)
    proba = model.predict_proba(test_imp[FEATURES])[:, 1]
    preds = model.predict(test_imp[FEATURES])
    preds_series = pd.Series(preds, index=test_idx)

    multi_class = y_test.nunique() > 1
    auc = roc_auc_score(y_test, proba) if multi_class else np.nan
    acc = accuracy_score(y_test, preds)
    pr_auc = average_precision_score(y_test, proba) if multi_class else np.nan

    is_early = metadata.loc[test_idx, "stage"].isin(EARLY_STAGES)
    early_idx = test_idx[is_early.values]
    early_recall = recall_score(y_test.loc[early_idx], preds_series.loc[early_idx]) if len(early_idx) > 0 else np.nan

    return {"auc": auc, "acc": acc, "pr_auc": pr_auc, "early_recall": early_recall}
```

---

## Section 9 — Two-Imputer Comparison

**Process:** each of the three CV schemes' actual train/test splits are computed **once** and reused for both imputers, so a performance gap is attributable to the imputer, not to different folds. For repeated CV this means materializing `RepeatedStratifiedKFold`'s 50 splits into a list up front rather than calling `.split()` separately per imputer; cohort-out and leave-one-site-out are already deterministic (fixed by `patient_cohort`/`sample_origin`), so identical splits come for free, but are still defined once for symmetry.

### 9a. Repeated Stratified 5-Fold × 10-Repeat CV — Both Imputers, Same Folds

`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`, stratified on `target_binary`. This is the main performance estimate — the only scheme with enough repeats (50 folds) to average out fold-to-fold noise.

| Imputer | AUC | PR-AUC | Early recall | Accuracy |
|---|---|---|---|---|
| KNN | 0.891 ± 0.027 | 0.790 ± 0.048 | 0.641 ± 0.097 | 0.810 ± 0.032 |
| **MICE** | **0.908 ± 0.023** | **0.841 ± 0.041** | **0.722 ± 0.104** | **0.842 ± 0.027** |

### 9b. Cohort-Out Split — Both Imputers, Same Split

Train on `patient_cohort == 'Cohort1'`, test on `'Cohort2'`. A **single** split (n=258 test), not an average over repeats — the noisiest of the three estimates here.

| Imputer | AUC | PR-AUC | Early recall | Accuracy |
|---|---|---|---|---|
| **KNN** | **0.908** | 0.629 | **0.714** | 0.864 |
| MICE | 0.837 | 0.629 | 0.679 | 0.868 |

### 9c. Leave-One-Site-Out — Both Imputers, Same Splits

For each `sample_origin` (BPTB, ESP, LIV, UCL), train on the other three sites, test on the held-out one — the strictest generalization test.

| Imputer | Site | n test | AUC | PR-AUC | Early recall | Acc |
|---|---|---|---|---|---|---|
| KNN | BPTB | 409 | 0.809 | 0.526 | 0.676 | 0.746 |
| KNN | ESP | 29 | 0.775 | 0.935 | 0.875 | 0.793 |
| KNN | LIV | 132 | 0.829 | 0.899 | 0.561 | 0.667 |
| KNN | UCL | 20 | NaN | NaN | NaN | 0.700 |
| MICE | BPTB | 409 | 0.842 | 0.629 | 0.757 | 0.770 |
| MICE | ESP | 29 | 0.783 | 0.943 | 1.000 | 0.793 |
| MICE | LIV | 132 | 0.834 | 0.897 | 0.667 | 0.727 |
| MICE | UCL | 20 | NaN | NaN | NaN | 0.850 |

UCL's 20 samples are all Benign (0 Control, 0 PDAC — a known EDA finding), so AUC/PR-AUC/early-recall have no meaning there; reported as `NaN` rather than a crash or a silently wrong number. Aggregate (mean across BPTB/ESP/LIV, excluding UCL):

| Imputer | AUC | PR-AUC | Early recall | Accuracy |
|---|---|---|---|---|
| KNN | 0.805 ± 0.027 | 0.786 ± 0.227 | 0.704 ± 0.159 | 0.726 ± 0.055 |
| **MICE** | **0.820 ± 0.032** | **0.823 ± 0.170** | **0.808 ± 0.172** | **0.785 ± 0.051** |

### 9d. Comparison Table and Deltas

Every `(scheme, imputer)` result in one place, with MICE-minus-KNN deltas computed explicitly:

| Scheme | Δ AUC | Δ PR-AUC | Δ Early recall | Δ Accuracy |
|---|---|---|---|---|
| Repeated 5×10 CV | +0.0174 | +0.0513 | +0.0810 | +0.0312 |
| Cohort-out | -0.0709 | -0.0001 | -0.0357 | +0.0039 |
| Leave-one-site-out | +0.0150 | +0.0367 | +0.1038 | +0.0587 |

---

## Final Summary

- **Outputs:** `feature_matrix` (590×7, `plasma_CA19_9` still raw/`NaN` where missing pre-fold), `METADATA` (590×5), `TARGETS` (590×2) — three frames aligned by index, never merged.
- **`plasma_CA19_9` is filled only inside `run_fold()`**, fold-by-fold, by a fresh imputer that never sees a fold's own held-out rows during fitting and never sees the target labels — every metric above is a leakage-free estimate.
- **Two imputers were built and compared empirically, not one picked by assumption** (Section 9) — the comparison ran three validation schemes twice each, on identical folds, and produced a clear, evidence-based answer.

### Verdict: **`MICE_CA19_9Imputer`** — recommended to carry forward.

**Reasoning:** MICE wins on all four metrics in the repeated 5×10 CV (the pipeline's main performance estimate, and the only scheme with enough repeats to average out fold-to-fold noise) and again on all four metrics in leave-one-site-out (the strictest generalization test). The PR-AUC and early-stage-recall gaps matter most for this specific problem — PR-AUC because the classes are imbalanced, and early-stage recall because catching PDAC early is the actual clinical point of a screening tool: MICE catches roughly 8 percentage points more early-stage cases in repeated CV and 10 points more in leave-one-site-out.

KNN only wins in the cohort-out scheme, and only clearly on AUC (0.908 vs. 0.837) — but that scheme is a *single* train/test split, not an average over repeats, so it's the noisiest of the three estimates; on that same scheme, PR-AUC is a near-exact tie (0.629 vs. 0.629) and early recall only mildly favors KNN. One volatile single-split result isn't enough to override two multi-fold schemes that both favor MICE by consistent, clinically-relevant margins. This is specific to this dataset size and predictor set — if either changed materially, the comparison should be rerun rather than assumed to still hold.

## Open Judgment Calls (flagged for confirmation, not yet locked in)

1. **Early-stage recall's stage cutoff** (Section 8) — `I`/`IA`/`IB`/`II`/`IIA`/`IIB` = early, `III`/`IV` = late; a resectability-based cut, not a canonical clinical threshold from the EDA. Easy to swap for a different boundary (e.g. I/II only).
2. **`METADATA['stage']` uses the sentinel-filled value**, not raw `NaN` (Section 6) — not explicitly specified in the build requirements, chosen as the more useful default for a non-modeling column.
3. **`sex` encoding is `F=0, M=1`** (Section 5) — arbitrary, just needs to stay consistent wherever this pipeline's output is consumed downstream.

**Resolved, no longer open:** the original KNNImputer-vs-regression-imputer judgment call (Section 7) — resolved empirically in Section 9 rather than left as an assumption. `MICE_CA19_9Imputer` is the recommended imputer per the Verdict above.

## Environment Note

`fyp_env`'s `python.exe` launcher is currently broken — its `pyvenv.cfg` points at a base Python install path from a different Windows user profile, so it cannot be invoked directly. The notebook above was executed end-to-end for verification using the system Python interpreter with `PYTHONPATH` pointed at `fyp_env`'s `site-packages` (which itself has the correct project dependencies — pandas 3.0.3, numpy 2.4.4, scikit-learn 1.9.0, xgboost 3.3.0 — installed and intact). Recreating `fyp_env` properly is recommended so normal `jupyter`/IDE kernel selection works again; this is an environment issue, not a pipeline defect.

## Helper Functions and Classes Added

| Name | Purpose |
|---|---|
| `KNN_CA19_9Imputer` | Fold-safe KNN imputer for `plasma_CA19_9` — `fit`/`transform` pattern, predictors hardcoded to the five non-target variables |
| `MICE_CA19_9Imputer` | Fold-safe `IterativeImputer`(`BayesianRidge`) imputer for `plasma_CA19_9` — same interface and fold-safety rules as KNN |
| `run_fold` | Shared, imputer-agnostic per-split routine: fit `imputer_class()` on train only, transform both sides, fit/score `XGBClassifier`, return AUC/accuracy/PR-AUC/early-stage recall |
| `EARLY_STAGES` | Constant set defining the early-vs-late stage cutoff used for early-stage recall (judgment call #1 above) |
