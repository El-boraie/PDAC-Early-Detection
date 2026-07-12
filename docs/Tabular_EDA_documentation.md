# Tabular EDA — Process, Code, and Results

**Source notebook:** `notebooks/Tabular EDA.ipynb`
**Dataset:** Debernardi et al. 2020 urinary biomarker dataset — 590 patients, 3 diagnosis classes (Control/Benign/PDAC)
**Data dictionary:** `data/raw/Debernardi et al 2020 documentation.csv` — this notebook stands on the data and this dictionary alone, not the written IR.
**Kernel / environment:** Python 3.13.12, venv at `C:\FYP\fyp_env`
**Run date:** 2026-07-08.

> Draft note: every number below is from the actual executed run, not estimated. Structure mirrors the notebook's own Section A-K headings (H2) and per-cell sub-headings (H3) so the two stay easy to cross-reference.

**Governing rule for this whole notebook: no transforms.** Biomarkers are lab test results — skew and outliers are expected real biology, not data errors. This EDA shows and quantifies distribution shape; it never log-transforms, winsorizes, or removes anything. Any earlier log1p views have been removed, not extended.

---

## Section A — Setup & Overview

**A1. Imports & Setup:** loads the CSV, builds the `diagnosis_label` map (1→Control, 2→Benign, 3→PDAC), sets plot style. One real bug fixed here: `DATA_PATH`/`OUT_DIR` originally pointed at `C:\FYP\IR\EDA\...`, which no longer exists after the repo scaffolding — repointed to `data/raw/` and `notebooks/tabular_eda_outputs/`.

**A2. Dataset Overview:** shape, dtypes, `head()`, `describe()`. Result: 590 rows, 15 columns (14 raw + `diagnosis_label`).

**A3. Helper Functions:** `values_by_diagnosis(df, col)` (NaN-dropped values split by diagnosis class) and `missing_count_by_group(df, col, group_col)` — pulled out once since both patterns repeat across nearly every later section. Also imports `scipy.stats.kruskal` here — not a new project dependency (`scipy` is already in `requirements.txt`), just a new import within this notebook.

---

## Section B — Initial Content Review (simple pass)

**B1. Unique Values Per Categorical Column:** a first, no-judgment look at `sex`, `patient_cohort`, `sample_origin`, `diagnosis`, `stage`, `benign_sample_diagnosis`.

**Result:**
- `sex`: `['F', 'M']`
- `patient_cohort`: `['Cohort1', 'Cohort2']`
- `sample_origin`: `['BPTB', 'ESP', 'LIV', 'UCL']`
- `diagnosis`: `[1, 2, 3]`
- `stage`: `['I', 'IA', 'IB', 'II', 'IIA', 'IIB', 'III', 'IV']`
- `benign_sample_diagnosis`: ~48 distinct free-text values (e.g. `Pancreatitis`, `Pancreatitis (Chronic)`, `Gallstones`, down to one-off entries like `Ill defined lesion in uncinate process`)

**B2. Value Counts, One Bar Chart Per Column (6 separate figures):** `unique_counts_sex.png`, `_patient_cohort.png`, `_sample_origin.png`, `_diagnosis.png`, `_stage.png`, `_benign_sample_diagnosis.png`. The `benign_sample_diagnosis` chart is deliberately shown in full (not capped to top-N) — the visual density of ~48 bars is itself the finding: this field is genuinely heterogeneous free text, not a clean small vocabulary like the other five columns.

**Bug fixed along the way:** pandas' string dtype leaves `NaN` as an actual float even after `.astype(str)` on an index (`counts.index.astype(str)` failed inside matplotlib's category converter). Fixed by using `[str(x) for x in counts.index]` instead, which correctly stringifies `NaN` → `'nan'` for the bar-chart x-axis. Affected all 6 B2 cells.

---

## Section C — Data Quality & Integrity Checks (detailed pass)

**C1. Duplicate Patient ID Check.** Result: **0 duplicate `sample_id` values** — all 590 unique.

**C2. Categorical Values vs. Data Dictionary:** compares B1's observed values against what the dictionary documents. Comparison sets are transcribed by hand from the dictionary's `Details` column (not auto-parsed — that column uses three different text conventions across rows: colon-separated for `sample_origin`, comma-separated for `patient_cohort`, `=`-separated for `sex`; a generic parser would be more fragile than citing the source directly).

**Result:** `patient_cohort`, `sample_origin`, `sex`, `diagnosis` — exact match, observed equals documented for all four.

**`stage` — a real, reportable discrepancy:**
- Documented (dictionary): `{IA, IB, IIA, IIIB, III, IV}`
- Observed (data): `{I, IA, IB, II, IIA, IIB, III, IV}`
- **Found but not documented:** `I`, `II`, `IIB`
- **Documented but never found:** `IIIB`

Most parsimonious read: `IIIB` in the dictionary is very likely a typo for `IIB` — `IIB` appears 68 times in the data (the second-most-common stage) while `IIIB` appears zero times. Not asserted as fact, just the most likely explanation given the pattern. **Action needed before using `stage` downstream: resolve this against the dictionary or the original study.**

`benign_sample_diagnosis` has no documented fixed vocabulary at all (dictionary describes it as free text) — validated structurally instead: non-null exclusively for Benign patients (183/183 Control null, 208/208 Benign non-null, 199/199 PDAC null — **confirmed exact**, no exceptions).

**C3. Numeric Range Sanity Checks.** Age: 26–89 years, 0 implausible (checked against a broad 0–120 bound; the dictionary gives no explicit clinical bound). All 6 biomarkers: 0 negative values (`plasma_CA19_9` min=0.0, `creatinine` min=0.057, `LYVE1` min=0.0001, `REG1B` min=0.001, `TFF1` min=0.005, `REG1A` min=0.0) — all physically valid concentrations.

---

## Section D — Missing Value Analysis

**D1. Missing Count/% Per Column.** `stage` 391 (66.3%), `benign_sample_diagnosis` 382 (64.8%), `REG1A` 284 (48.1%), `plasma_CA19_9` 240 (40.7%). 11 columns with 0 missing.

**D2. Structural Missingness — Explicit Sentinel Categories.** `stage` and `benign_sample_diagnosis` missingness is **not a defect** — C2 confirmed 100% of `stage` nulls are non-PDAC and 100% of `benign_sample_diagnosis` nulls are non-Benign, exactly. Because that exclusivity is exact (verified, not assumed), a plain `.fillna()` on the whole column is equivalent to conditioning on diagnosis — no masking logic needed.

```python
df['stage'] = df['stage'].fillna('No Cancer')
df['benign_sample_diagnosis'] = df['benign_sample_diagnosis'].fillna('Control/PDAC')
```

**Result:** `stage` now has a `No Cancer` category (391 rows) sitting alongside the 8 real stage codes; `benign_sample_diagnosis` now has 382 `Control/PDAC` rows. Zero nulls remain in either column. This adds explicit categories rather than imputing a statistic — "not applicable" isn't a value that can be estimated.

**D3. Missingness of `plasma_CA19_9`/`REG1A` by Diagnosis Class.** `plasma_CA19_9`: Control 49.7%, Benign 48.1%, PDAC 24.6% missing. `REG1A`: Control 56.8%, Benign 58.2%, PDAC 29.6% missing.

**D4. Same, by Cohort and Sample Origin — the sharpest finding in this notebook:**
- **`REG1A` is missing in 258/258 (100%) of Cohort2 patients**, vs only 26/332 (7.8%) of Cohort1. Not "assessed in fewer patients" in a spread-out sense — REG1A is essentially absent from an entire cohort.
- By origin, `REG1A` missingness concentrates in BPTB (63.3%) and ESP (65.5%), while LIV (4.5%) and UCL (0%) have near-complete coverage.
- `plasma_CA19_9` missingness is comparatively balanced across cohorts (36.1% Cohort1 vs 46.5% Cohort2) and more evenly spread by origin too, except ESP and UCL are both 100% missing there.

**Practical consequence:** imputing `REG1A` needs to account for this cohort/site structure — a single global imputation statistic would be wrong, since "missing" for Cohort2 means "this entire cohort doesn't have this feature."

---

## Section E — Class Distribution & Demographics

**E1. Target Class Distribution:** Control 183 (31.0%), Benign 208 (35.3%), PDAC 199 (33.7%) — reasonably balanced three-way split.

**E2. Cohort & Sample Origin Crosstabs, with discussion:**
- **Cohort1 has a dramatically higher PDAC rate than Cohort2**: 162/332 (48.8%) vs 37/258 (14.3%). Since `patient_cohort` distinguishes "previously used samples" from "newly added samples" per the dictionary, this may simply reflect how each batch was originally selected, not a biological cohort difference. Matters for any cohort-based split.
- **UCL contributes only Benign cases**: all 20 UCL samples are Benign (0 Control, 0 PDAC) — a representativeness gap, not a data error.

**E3. Age & Sex Demographics.** PDAC patients skew older (mean 66.2 vs 54.7 Benign / 56.3 Control) and more male (58.3% vs 51.4% Benign / 37.2% Control).

---

## Section F — Distribution Shape (Skewness) — NO TRANSFORM APPLIED

**F1. Skewness — All 6 Biomarkers, one table, NaNs dropped:**

| Biomarker | Skewness | Interpretation |
|---|---|---|
| creatinine | 1.47 | highly skewed |
| LYVE1 | 1.39 | highly skewed |
| REG1B | 3.33 | highly skewed |
| TFF1 | 5.16 | highly skewed |
| REG1A | 4.47 | highly skewed |
| plasma_CA19_9 | 8.02 | highly skewed |

**The IR's own headline claim is directly contradicted here**: IR states skewness of 10.37 for `plasma_CA19_9`; the real computed value is **8.02**. Not merely "unbacked by the notebook" (the original gap) — actually a different number once computed.

**F2. Raw Histograms, one figure per biomarker (6 total):** `hist_raw_creatinine.png` through `hist_raw_plasma_CA19_9.png`, each split by diagnosis class. Raw values only — no log1p view, per this notebook's no-transform policy. `plasma_CA19_9` visibly shows the extreme right-skew consistent with its skewness of 8.02 — a handful of PDAC patients reach ~30,000 U/ml against a bulk of the distribution under 1,000.

**F3. Boxplots, one figure per biomarker (6 total):** `boxplot_creatinine.png` through `boxplot_plasma_CA19_9.png`, same split, same no-transform policy.

---

## Section G — Outlier Quantification (IQR)

**G1. Q1/Q3/IQR and outlier count/% per biomarker, per diagnosis class — purely descriptive, nothing removed or capped.**

Total outliers per biomarker (summed across all 3 classes): REG1B 62, plasma_CA19_9 38, TFF1 35, REG1A 30, LYVE1 24, creatinine 24.

Full per-class breakdown (18 rows: 6 biomarkers × 3 classes) is in the notebook's G1 cell output — notable extremes: `plasma_CA19_9` PDAC upper bound is 3,494.2 U/ml (13/150 patients, 8.7%, exceed it) and `REG1A` PDAC upper bound is 3,360.5 ng/ml (10/140, 7.1%). These are documented, not removed.

---

## Section H — CA19-9 Clinical Deep Dive

**H1. Cutoff Analysis at 37 U/ml.** Log-transformed panel **removed** per the no-transform policy — now a 2-panel figure (raw boxplot + clinical-threshold bar chart), not the original 3-panel version.

**Result:** CA19-9 missingness by class — Control 49.7%, Benign 48.1%, PDAC 24.6% missing (PDAC patients disproportionately more likely to actually have this blood test run — plausibly because PDAC diagnosis prompts more complete clinical workup).

---

## Section I — Correlation & Statistical Testing

**I1. Pearson Correlation Heatmap & Ranked Bar Chart.** Ranked by |r| with diagnosis: LYVE1 0.54, TFF1 0.39, REG1B 0.38, age 0.31, plasma_CA19_9 0.26, REG1A 0.26, creatinine 0.07.

**I2. Kruskal-Wallis H-Test** — non-parametric alternative to ANOVA, appropriate given Section F confirms these biomarkers are far from normal (Pearson assumes a linear relationship they don't have).

| Feature | H-statistic | p-value |
|---|---|---|
| plasma_CA19_9 | 209.01 | 4.1e-46 |
| LYVE1 | 208.88 | 4.4e-46 |
| TFF1 | 161.86 | 7.1e-36 |
| REG1B | 133.21 | 1.2e-29 |
| age | 92.11 | 1.0e-20 |
| REG1A | 44.25 | 2.5e-10 |
| creatinine | 1.11 | 0.57 (not significant) |

**Key finding: Kruskal-Wallis changes the ranking Pearson gives.** `plasma_CA19_9` ranks 5th by Pearson (0.264) but is essentially tied for **1st** by Kruskal-Wallis (209.01 vs LYVE1's 208.88) — Pearson was understating CA19-9's real discriminative power because its relationship with diagnosis is highly nonlinear (consistent with its skewness of 8.02), exactly the scenario Kruskal-Wallis exists to catch. `creatinine` is confirmed not significant by both methods (Pearson 0.07, Kruskal-Wallis p=0.57) — consistently the weakest predictor either way.

---

## Section J — PDAC Stage Analysis

**J1. Stage Distribution & Median Biomarkers by Stage** (PDAC patients only, 199 total). Stage III most common (76), then IIB (68), IV (21), IB (12), IIA (11), II (7), IA (3), I (1). Median CA19-9 generally rises with stage (11.0 at IA → 941.0 at IV), though not perfectly monotonic (IIB's 267.0 dips slightly below III's 553.0 in the ordering used).

---

## Section K — EDA Summary

**K1. Findings Summary**, pulling real numbers from F1 (skewness), G1 (IQR outliers), D2/D4 (missingness framing), and I2 (Kruskal-Wallis) — explicitly states that skewness and outliers were **documented, not transformed**, since these are genuine lab test results. Full text is the notebook's own final cell; the preprocessing-requirements list it prints has been updated to remove "log-transform skewed biomarkers" (no longer a stated requirement — direct contradiction of this notebook's governing rule) and instead notes that outlier handling for modeling purposes is a downstream decision this EDA does not make.

---

## Bugs found and fixed during this rebuild (not analysis findings, code correctness)

1. **Dead path**: `DATA_PATH`/`OUT_DIR` pointed at `C:\FYP\IR\EDA\...`, which doesn't exist post-scaffolding. Repointed to `data/raw/` and `notebooks/tabular_eda_outputs/`.
2. **matplotlib 3.11 API change**: `boxplot(..., labels=[...])` was removed in favor of `tick_labels=[...]` — affected 3 cells in an earlier pass of this notebook, all fixed.
3. **pandas string-dtype NaN handling**: `.astype(str)` on an Index containing `NaN` leaves it as a float, not a string, causing a `TypeError` inside matplotlib's category converter for any categorical bar chart including a `NaN` category (`stage`, `benign_sample_diagnosis` before Section D2's fillna). Fixed with `[str(x) for x in counts.index]`.

None of these three were caught by code review alone — all surfaced only once the notebook was actually executed top-to-bottom, which is why that execution step matters more than reading the code.
