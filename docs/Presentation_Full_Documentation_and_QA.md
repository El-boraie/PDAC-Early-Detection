# PDAC Multimodal Risk Detection — Full Presentation Reference

**Purpose of this document:** a single, self-contained reference for building presentation
slides — covers both model-building branches (clinical/tabular and imaging/CT), evaluation
results, explainability, and fusion, plus a large bank of anticipated "why" questions with
answers you can speak to directly. Every number here is pulled from the project's own
executed notebooks and `docs/*.md` files, not estimated.

**Source documents this was compiled from:** `docs/Tabular_Preprocessing_documentation.md`,
`docs/Tabular_Model_Comparison_documentation.md`, `docs/Tabular_Final_Fit_documentation.md`,
`docs/Clinical_SHAP_documentation.md`, `docs/Tabular_EDA_documentation.md`,
`docs/CT_EDA_documentation.md`, `docs/CT_Preprocessing_documentation.md`,
`docs/Imaging_3Fold_Training_Results_documentation.md`,
`docs/Imaging_5Fold_Training_Results_documentation.md`,
`docs/Imaging_Evaluation_documentation.md`, `docs/Imaging_Confound_Check_documentation.md`,
`docs/Imaging_Session_Summary_2026-07-18.md`, `docs/Fusion_documentation.md`,
`docs/Fusion_Evaluation_documentation.md`, `docs/Dashboard_documentation.md`,
`docs/Tech_Stack_documentation.md`, `src/imaging/models.py`.

---

## 0. System Overview

**Project goal:** early PDAC (pancreatic ductal adenocarcinoma) risk detection from two
independent, unpaired data modalities:

1. **Tabular / clinical branch** — urinary biomarker panel → XGBoost binary classifier
   (PDAC vs. not-PDAC).
2. **Imaging / CT branch** — abdominal CT axial slices → ResNet50-U-Net joint
   segmentation + detection model.

3. **Fusion layer** — a fixed, hand-set rule (not a trained model) that combines each
   branch's *calibrated* probability into one score, used only at inference time by the
   dashboard. There is **no paired dataset** anywhere in this project (no patient has both
   a CT scan and a urine sample), so fusion cannot be trained or benchmarked jointly — this
   is a structural fact about the data, stated up front because it recurs throughout.

**Pipeline shape (both branches independently):** EDA → preprocessing → candidate
comparison (cross-validated) → final all-data fit + calibration → explainability →
disclosed limitations. The imaging branch adds a dedicated confound-check stage the
tabular branch didn't need (see §3.5).

**High-level architecture:**

```
Urinary biomarkers ──▶ MICE imputer ──▶ XGBoost ──▶ Platt calibration ──▶ tabular_proba ──┐
                                                                                            ├──▶ fused_score = 0.4·imaging + 0.6·tabular
CT scan (slices)  ──▶ ResNet50-U-Net ──▶ Platt calibration ──▶ mean-over-slices ──▶ imaging_proba ──┘
```

---

## 1. Datasets

| | Tabular / Clinical | Imaging / CT |
|---|---|---|
| **Source** | Debernardi et al. 2020 urinary biomarker dataset | MSD Task07 Pancreas (281 pts, cancer) + NIH Pancreas-CT / TCIA CT-82 (80 pts, healthy) |
| **N (patients)** | 590 | 361 (281 MSD + 80 NIH) |
| **N (rows/slices)** | 590 (one row per patient) | 90,693 axial slices (72,077 MSD + 18,616 NIH) |
| **Classes** | Control 183 / Benign 208 / PDAC 199 → binary target (391 not-PDAC / 199 PDAC) | Cancer (MSD, 100%) vs. Healthy (NIH, 100%) — **perfectly confounded with dataset/scanner of origin** (critical caveat, see §3.5) |
| **Features** | 7: `creatinine, LYVE1, REG1B, TFF1, plasma_CA19_9, age, sex` (REG1A dropped) | 2D axial slices, 320×320, HU-windowed [-150,+250] → normalized [0,1], 3-channel replicated for ResNet |
| **Missingness** | `plasma_CA19_9` missing in 240/590 (40.7%); `REG1A` missing in 284/590 (48.1%, and 100% of Cohort2) | NIH has no segmentation mask (image-only, healthy-negative addendum) |
| **Split strategy** | Repeated 5×20 stratified CV (main), cohort-out, leave-one-site-out | Patient-level, lesion-quartile-stratified 5-fold CV (never slice-level — would leak anatomy) |

**Why two completely separate datasets instead of one paired cohort?** No public dataset
exists with both CT imaging and this specific urinary biomarker panel for the same
patients — this is a real, structural data-availability constraint, not a design choice
that could have been avoided. It's the reason fusion has to be a fixed rule rather than a
trained joint model (§4).

---

## 2. Branch A — Tabular / Clinical Model

### 2.1 Preprocessing (`src/clinical/tabular_clean.ipynb`)

- **Target:** `diagnosis` (1=Control, 2=Benign, 3=PDAC) → binary `target_binary` (1=PDAC).
- **`REG1A` dropped entirely** — missing in 100% of Cohort2 (258/258) vs. 7.8% of Cohort1;
  an entire cohort never ran this assay, so imputing it would invent values for a whole
  cohort with zero ground truth, and a missingness indicator would just re-encode
  `patient_cohort`, which already exists as a column.
- **`plasma_CA19_9` kept, left as `NaN` at this stage** — it's the single feature with real
  missingness that carries genuine signal (3rd-highest Pearson r=0.26, joint-highest
  Kruskal-Wallis H-statistic among all six biomarkers per the EDA), so it's imputed
  fold-by-fold instead of dropped.
- **No log transform, no winsorizing, no outlier removal, no scaling of the final feature
  matrix, anywhere** — these are lab test results; skew and outliers are real disease
  signal. XGBoost splits on rank order regardless of scale, so transforms would only
  destroy information for zero benefit.
- **`sex` encoded 0/1** (F=0, M=1 — arbitrary but fixed); no other column transformed.
- **Final feature matrix:** `(590, 7)`.

### 2.2 Imputer Selection — KNN vs. MICE (empirical, not assumed)

Two fold-safe imputer classes were built for `plasma_CA19_9` only (never zero-filled — 0 is
a real physiological value; never fit on the full dataset first; never predicts from
`diagnosis`/target — that would leak the label):

- **`KNN_CA19_9Imputer`**: `StandardScaler` + `KNNImputer(n_neighbors=5)` on the 5
  non-target predictors.
- **`MICE_CA19_9Imputer`**: `IterativeImputer(estimator=BayesianRidge())` — a regression
  model of CA19-9 on the other 5 predictors, refined iteratively.

**Result — MICE wins on all 4 metrics in the two multi-fold schemes:**

| Scheme | Δ AUC (MICE−KNN) | Δ PR-AUC | Δ Early recall | Δ Accuracy |
|---|---|---|---|---|
| Repeated 5×10 CV | **+0.0174** | **+0.0513** | **+0.0810** | **+0.0312** |
| Cohort-out (single split) | −0.0709 | −0.0001 | −0.0357 | +0.0039 |
| Leave-one-site-out | **+0.0150** | **+0.0367** | **+0.1038** | **+0.0587** |

KNN only wins cohort-out, and that's a *single* train/test split (the noisiest of the three
estimates) — one volatile result doesn't override two multi-fold schemes both favoring
MICE by clinically meaningful margins (8–10 points of early-stage recall).

### 2.3 Model Comparison — XGBoost vs. Logistic Regression vs. Random Forest

Three CV schemes, **identical folds/seeds across all three models** so any performance gap
is attributable to the model, not to different data landing in different folds.

**Repeated stratified 5-fold × 20-repeat CV (main estimate, 100 folds):**

| Model | AUC | Precision | Recall (Sensitivity) | Specificity | F1 | Early Recall |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.9077 ± 0.0224 | 0.7760 | **0.7433** | 0.8884 | **0.7573** | **0.6379** |
| Logistic Regression | 0.8786 ± 0.0304 | 0.7877 | 0.6157 | 0.9130 | 0.6884 | 0.5168 |
| Random Forest | 0.9068 ± 0.0239 | 0.7943 | 0.6920 | 0.9060 | 0.7370 | 0.6109 |

**Cohort-out (single split):** Random Forest wins every metric (AUC 0.8901, recall 0.7568).
**Leave-one-site-out (strictest generalization test):** XGBoost wins AUC-adjacent metrics
that matter most — recall 0.7783 vs. RF 0.7290 vs. LR 0.7483.

### Verdict: **XGBoost**

- Best recall/sensitivity — the clinically prioritized metric, since missing a PDAC case
  is far costlier than a false alarm — in **2 of 3 schemes** (main repeated CV and the
  strictest leave-one-site-out test).
- AUC is essentially tied with Random Forest in those same two schemes (0.9077 vs. 0.9068,
  gap of 0.0009 — negligible), so this isn't "picking RF's AUC lead over XGBoost's recall
  lead" — XGBoost doesn't have an AUC deficit worth trading against.
- Random Forest only wins clearly in cohort-out, a single non-repeated split — the
  statistically weakest of the three estimates.
- Hyperparameters: `n_estimators=100, max_depth=3` (fixed by the project spec, not
  independently tuned per model — flagged as a real caveat, not hidden).

### 2.4 Final Fit (`src/clinical/clinical_final_fit.ipynb`)

- Refits **only** XGBoost (the settled winner) and MICE (the settled imputer) on **all 590
  patients**, no folds — validation is already done; this step answers "what's the single
  best deployable version, using every patient available?"
- **Calibration: Platt scaling** (1-D logistic regression on raw score → probability),
  chosen over isotonic regression because the *effective* sample size is 590 unique
  patients (not the 11,800 OOF rows), and isotonic typically needs thousands of samples to
  avoid staircase overfitting.
- Fit on **all 11,800 saved out-of-fold predictions** from the comparison stage (590
  patients × 20 CV repeats) — never on the final model's own in-sample predictions, which
  would be circular (it has already seen every one of those labels).
- **In-sample sanity check (NOT a performance claim):** AUC=1.0, recall=1.0 — expected
  symptom of scoring a model on data it memorized; shown only next to the real validated
  numbers (0.9077 / 0.7433) so the gap is impossible to miss.
- Artifacts: `model.pkl`, `ca19_9_imputer.pkl`, `calibrator.pkl`, `model_card.json` in
  `checkpoints/clinical/final/` — all four use **generic filenames**, never naming the
  winning model type, so the deployed app never has to branch on which model won.

### 2.5 Explainability — SHAP

- **`TreeExplainer`** (exact Shapley values via tree traversal), not `KernelExplainer`
  (model-agnostic Monte Carlo approximation) — exact and free beats approximate for a tree
  model.
- Explains **`X_imputed`** (what the model actually sees at inference), not the raw
  NaN-containing matrix.
- **Raw log-odds (margin) space**, not calibrated probability — SHAP's additivity
  guarantee (`expected_value + sum(shap) == model output`) only holds in the space the
  trees themselves produce; the Platt calibrator sits after the raw model as a separate
  squashing function.
- **Additivity check: PASS, max abs error 4.77e-06** (threshold 1e-4) — confirms SHAP
  values reconstruct the model's real output, not an approximation drifting from it.
- All 590 patients, in-sample — this is a "what did the model learn?" question, not a
  generalization question (already answered by CV).

**Global feature ranking (mean |SHAP|, log-odds space):**

| Rank | Feature | mean(\|SHAP\|) | Note |
|---|---|---|---|
| 1 | **LYVE1** | 1.669 | Strongest driver — an empirical finding, not assumed |
| 2 | **plasma_CA19_9** | 1.293 | The clinically familiar biomarker — 2nd here, compressed by imputation (see caveat) |
| 3 | creatinine | 0.832 | General renal-function covariate |
| 4 | age | 0.810 | Demographic risk factor |
| 5 | TFF1 | 0.599 | Urinary marker |
| 6 | REG1B | 0.382 | Urinary marker |
| 7 | sex | 0.073 | Weakest driver |

**Two caveats to state explicitly in the presentation:**
1. **Scale dependency:** ranking is on log-odds, not probability. A second check in
   probability space gives Spearman ρ=0.9643 vs. the log-odds ranking — only LYVE1 and
   CA19-9 swap places (CA19-9 edges ahead in probability terms). So *which* feature is "the
   top one" genuinely depends on which scale you ask in — worth saying out loud rather than
   picking one number to quote.
2. **Imputation confound:** 240/590 (41%) of `plasma_CA19_9` values are MICE-imputed, not
   measured. Global importance for CA19-9 is a mix of real biomarker signal (350 measured
   rows) and imputer-model behavior (240 imputed rows) — e.g. one waterfall example (row
   576) shows an *imputed* CA19-9 value as the single largest driver of a PDAC
   classification, which would be a wrong read if presented as "measured CA19-9 drove
   this."

---

## 3. Branch B — Imaging / CT Model

### 3.1 Preprocessing (`src/imaging/imaging.ipynb`)

| Step | Choice | Why |
|---|---|---|
| Resampling | Isotropic 1×1×1mm, linear (image) / nearest-neighbour (mask) | Raw Z-spacing varied 0.7–7.5mm (>10× range) — needed a common grid; nearest-neighbour on masks avoids inventing fractional labels |
| HU window | Clip [-150, +250], normalize to [0,1] | Clinical reference ranges: pancreas 25–55 HU, tumour 10–40 HU |
| Slice extraction | 2D axial, along Z | 3D was infeasible on the 4GB local GPU (see Tech Stack) |
| Risk score (per slice) | `tumour_px / (pancreas_px + tumour_px)`, floored to 0 below 50 combined px | Organ-relative ratio is learnable; raw `tumour_px/total_px` is background-dominated and near-zero everywhere — rejected |
| Storage | uint8 `.npy`, 4× smaller than float32 | Kept 25.35GB actual vs. 50GB disk budget |
| Split | **Patient-level**, lesion-size-quartile stratified, 70/15/15 → later 5-fold | Splitting at slice level would leak the same patient's anatomy across train/test |

**Result:** 281/281 MSD patients passed integrity checks (0 excluded), 72,077 slices.

**NIH healthy-negative addendum:** MSD alone is 100% cancer patients — no true
healthy-negative examples for the detection head to learn presence/absence from. Added 80
NIH patients (image-only, no mask — the NIH multi-organ auto-segmentations have no usable
label key). Result: 90,693 total slices across 361 patients, merged into a `StratifiedKFold`
(k=5) split by class, so every fold contains both cancer and healthy patients.

### 3.2 Model Architectures — Two Candidates

**`baseline_cnn`** (97,761 params) — from-scratch CNN, detection head only, no segmentation.

**`resnet50_unet`** (71,876,484 params) — the winner. Pretrained ImageNet ResNet-50 encoder
+ U-Net decoder (segmentation: background/pancreas/tumour) + a detection head off the
encoder bottleneck (`layer4`, 2048ch → `AdaptiveAvgPool2d(1)` → `Linear(2048,1)`). One
shared encoder, two task heads — segmentation and detection are trained jointly. Skip
connections from each encoder stage feed the corresponding decoder `UpBlock` (upsample →
concat → conv/BN/ReLU ×2), standard U-Net structure.

**Why this architecture pairing?** `baseline_cnn` is the cheap/interpretable control that
tests whether a from-scratch model is competitive; `resnet50_unet` tests whether
transfer-learned features + a segmentation objective help. Only `resnet50_unet` can produce
the segmentation mask the project's risk-score (tumour/gland pixel ratio) structurally
depends on — `baseline_cnn` cannot support that downstream feature regardless of its
detection accuracy.

### 3.3 Training Results

**Hardware note (relevant to a "why" question):** local RTX 3050 (4GB) measured at 37.81
min/epoch, compute-bound at 97% GPU utilization — too slow for the schedule. Training moved
to a rented RunPod RTX 6000 Ada (48GB VRAM, ~$0.77/hr). Total imaging training spend:
~$3–4.

**3-fold → 5-fold:** the original `FOLDS_TO_RUN=[0,1,2]` cap was a schedule constraint from
the local-GPU era, not methodological. On the rented GPU, 2 extra folds cost ~27 minutes and
under $1, so the cap was lifted — the 5-fold run gives every one of 361 patients exactly one
out-of-fold prediction (181,386 OOF rows), vs. only 60.7% patient coverage at 3 folds.

**5-fold results (pre-augmentation-fix numbers — see §3.5 for the currently-promoted,
augmented version):**

| Model | ROC-AUC | Recall | Precision | Specificity | F1 | Dice | IoU |
|---|---|---|---|---|---|---|---|
| **resnet50_unet** | **0.9938** | **0.9833** | 0.9644 | 0.8614 | **0.9736** | 0.395 | 0.282 |
| baseline_cnn | 0.9819 | 0.8845 | 0.9538 | 0.8141 | 0.9110 | — (no seg head) | — |

**Winner: `resnet50_unet`.** ROC-AUC gap alone (0.012) would be inconclusive, but recall
diverges sharply (98.3% vs. 88.5%, ~10pt gap) and — just as importantly — `resnet50_unet`'s
recall is far more **stable** across folds (95.9–99.7% band) vs. `baseline_cnn`'s wild
swings (68.9%–99.7%). Concretely, in one fold `baseline_cnn`'s specificity collapses to
0.446 (flags almost everything as cancer); in another its recall collapses to 0.689 (misses
~31% of actual cancer slices). This fold-to-fold instability is itself an argument against
`baseline_cnn` for a clinical pipeline, not just a mean-metric loss. `baseline_cnn` does win
cleanly on precision/specificity — an honest, stated tradeoff (more conservative vs. more
sensitive).

### 3.4 Evaluation, Promotion, Calibration (`src/imaging/imaging_evaluation.ipynb`)

- **Candidate comparison** pulled from each model's own saved `model_card.json` (not
  recomputed) to avoid a subtly different number appearing for no reason.
- **Promotion:** `checkpoints/imaging/final/model.pt` is a separate, **all-data, no-folds**
  fit (mirrors the clinical branch's final-fit philosophy) — 361 patients, 90,693 rows, no
  held-out set.
- **Fixed epoch count instead of early stopping:** a no-folds fit has no validation set, so
  patience-based early stopping (used everywhere else) isn't available. Instead: the mean
  `best_epoch` across the 5-fold candidate run's per-fold values (`[4,0,2,12,8]`, mean 5.2)
  → rounded → **6 total epochs**.
- **In-sample sanity check (not a performance estimate):** mean P(cancer|true=cancer) =
  0.9963, mean P(cancer|true=healthy) = 0.0159 — sane, well-separated, non-degenerate.
- **Calibration:** Platt scaling, fit on all 90,693 OOF rows (361 unique patients) — same
  reasoning as the clinical branch (isotonic needs more effectively-independent samples
  than are really available here).

### 3.5 Explainability & the Confound Check — the Most Important "Why" Story in This Project

**The concern, precisely:** `dataset` (MSD vs. NIH) and `class` (cancer vs. healthy) are
**perfectly correlated** in this data — every MSD patient is cancer, every NIH patient is
healthy. A model could reach very high accuracy by learning "which scanner/institution
produced this image" instead of real pathology, and every standard metric (ROC-AUC, recall)
would look identical either way. This is why a dedicated confound-check investigation was
run — it's the only way in this project to tell the two explanations apart.

**Six rounds of investigation**, escalating from correlational to causal evidence:

| Round | Method | Finding |
|---|---|---|
| 1 | Grad-CAM alone | Alarming: enrichment 0.003x vs. chance, attention sits in image corners, not on the pancreas |
| 2 | + Integrated Gradients | Disagreement — IG showed enrichment ~1.08x (near chance), not corner-biased. Inconclusive alone. |
| — | Padding investigation | Confirmed real synthetic zero-padding exists from the BOX=320 packing step; top-padding differs significantly between MSD/NIH (p=0.011) — a plausible partial, non-anatomical cue |
| 3 | 6 attribution methods (4 gradient-based + EigenCAM gradient-free + IG input-space) | All 4 gradient-based CAM methods converge near-zero (~0.003–0.006x); removing gradients (EigenCAM) jumps to 0.640x; removing `layer4` entirely (IG) reaches ~1.08x (chance). Tentative read: "probably an attribution-method artifact" |
| **4** | **Occlusion sensitivity (causal, not inferred)** | **Reverses the Round 3 read.** Blanking the real tumour moves the prediction **significantly less** than blanking a random patch (0.47x control, p=0.0075); blanking the detected padding moves it **significantly more** (2.03x control, p=0.044). Causal evidence outranks gradient/activation inference. |
| **5** | Fix: random-resized-crop augmentation, retrain | **Padding shortcut fixed and verified**: 2.03x → 1.08x (p=0.917, statistically indistinguishable from control). Tumour under-reliance improved numerically (0.47x→0.62x) but **still significant** (p=0.0065) — not fixed. |
| 6 | Fix attempt #2: mask-preserving random erase, retrain | **Did not help** — tumour reliance unchanged (0.58x vs. 0.62x, still significant p=0.0148); padding-sensitivity point estimate rose to 2.86x though not statistically significant at this smaller n. **Reverted** — model restored to the verified Round 5 state. |

**Why occlusion (Round 4) overturned Round 3's gradient-based read:** every method through
Round 3 — even the "gradient-free" ones — only *infers* importance from gradients or
activations; none of them intervene on the input and observe the actual causal effect.
Occlusion physically replaces a region with a neutral fill (the image's own median
intensity) and re-runs the model — a real causal experiment, not an inference. A
methodological fix was needed mid-analysis too: the first occlusion run measured deltas in
*probability* space and found nothing, because baseline P(cancer) on correctly-classified
slices averages 0.9991 — the sigmoid is already saturated, so no local occlusion can move a
probability much regardless of true importance. Switching to **logit space** (unbounded,
doesn't saturate) is what produced the informative result.

**Current, disclosed status of the promoted model (`checkpoints/imaging/final/model.pt`):**
- **Padding shortcut: fixed and verified** (augmentation retrain, Round 5).
- **Tumour under-reliance: still open and unresolved** — the real tumour region moves the
  model's prediction significantly *less* than a same-sized random patch of ordinary
  tissue, even after two remediation attempts.
- **Practical consequence, quoted directly from `model_card.json`'s `known_limitations`:**
  detection metrics (ROC-AUC 0.993, recall 0.963) are strong and reproducible, but are
  **not yet confirmed evidence of tumour-specific pixel-level reasoning**. This caveat is
  deliberately carried through every downstream artifact that surfaces these numbers
  (fusion model card, fusion evaluation notebook, dashboard) — never presented standalone.

**Why this matters for the presentation:** this is the single strongest "we did rigorous
science, not just curve-fitting" story in the project. Leading with the raw AUC/recall
numbers without this caveat would misrepresent what's actually been demonstrated — say that
explicitly if asked "so is the imaging model good?"

---

## 4. Fusion

### 4.1 The Hard Constraint

**No patient in this project has both a CT scan and a urine sample.** MSD/NIH (imaging) and
Debernardi et al. (tabular) are three entirely separate, unpaired cohorts. There is
therefore **no ground truth against which a fused prediction could ever be scored** — not a
shortcut avoided, a structural fact about the available data. Consequently: no joint
precision/recall/F1/ROC-AUC/confusion matrix is ever computed anywhere in this project, and
the fusion weights below are **fixed, hand-set judgment calls**, not fitted parameters.

### 4.2 Rule 1 — Cross-Modality Weighting

```
fused_score = 0.4 × imaging_calibrated_proba + 0.6 × tabular_calibrated_proba
```

**Why 0.4/0.6, not 0.5/0.5 or something fitted:** imaging is weighted *below* tabular
because it carries the disclosed, unresolved confound-check finding (§3.5) — its detection
confidence is not yet confirmed to reflect genuine tumour-specific reasoning, so it's
deliberately not allowed to dominate a disagreement between branches. The discount is kept
*modest* (0.4, not lower) because imaging's own calibration is strong and, post-calibration,
slightly under-confident — a larger discount would over-correct for that.

### 4.3 Rule 2 — Imaging Slice-to-Volume Aggregation

```
patient_score = mean(per-slice calibrated probabilities across the whole scan)
```

**Why this was needed:** a real CT scan is a volume (128–526 slices in this dataset); the
model's native training/eval granularity is per-slice. Requiring a user to pre-select the
one slice showing the tumour would push the model's own detection job onto them.

**Why `mean`, empirically (not asserted):** measured against 7 candidate aggregation rules
on real 5-fold OOF predictions:

| Rule | ROC-AUC | F1 | Brier |
|---|---|---|---|
| **mean** | 0.9973 | **0.9893** | **0.0142** |
| median | 0.9975 | 0.9875 | 0.0146 |
| max | 0.9913 | 0.9605 | 0.0577 |
| top-5%-slice-mean | 0.9974 | 0.9689 | 0.0449 |

`mean` wins on F1 and Brier; `max`/top-k buy marginally higher recall but collapse
precision (~27% false-positive rate on healthy patients).

**The honest catch — why `mean` winning is itself diagnostic, not just good news:** a
within-patient spread check shows cancer patients' per-slice scores barely move (std≈0.05;
even the *lowest*-scoring slice in a cancer patient averages 0.42 probability). If the model
genuinely localized tumours slice-by-slice, within-patient variance would be high and
`max`/top-k would win instead. It doesn't. **`mean` outperforming the alternatives is itself
a fingerprint of the same confound** described in §3.5 — the "cancer signal" reads as global
to the whole volume, not localized to tumour-bearing slices. Volume aggregation improves the
numbers *without resolving*, and can visually obscure, the tumour-under-reliance finding.
This reasoning is recorded verbatim in the fusion model card, not just spoken about.

### 4.4 Evaluation — What Can and Cannot Be Claimed

**Pair comparison (each branch's own metrics, side by side, never combined):**

| | Imaging: `resnet50_unet` (slice-level, 5-fold CV) | Tabular: `XGBoost` (patient-level, repeated 5×20 CV) |
|---|---|---|
| Precision | 0.9920 | 0.7760 |
| Recall | 0.9628 | 0.7433 |
| ROC-AUC | 0.9934 | 0.9077 |

**These are NOT comparable as "which model is better"** — they come from two entirely
different evaluations (slice-level detection on CT vs. patient-level classification on
urine biomarkers), on two different cohorts.

**Calibration quality (Brier score, lower=better):**

| | Tabular | Imaging |
|---|---|---|
| Raw | 0.1213 | 0.0303 |
| Calibrated | 0.1189 | 0.0282 |

**Worked fusion examples are illustrative only, not a benchmark:** 5 rows built by taking
real raw OOF probabilities from each branch separately, calibrating each with that branch's
own real calibrator, then **artificially pairing** one imaging score with one tabular score
(shuffled — no real patient correspondence) purely to demonstrate the mechanism. No accuracy
number is computed for these, because none is possible.

**Interface guarantee:** if only one modality is supplied, `fused_score` is that branch's
own calibrated score, returned untouched — never silently degraded or combined with a
fabricated placeholder for the missing branch.

---

## 5. Dashboard (Deployment Layer, Brief)

- **Streamlit app**, 5 pages: Register → Predict → Analytics → Reports → About.
- Consumes only already-trained, already-calibrated artifacts under
  `checkpoints/{clinical,imaging,fusion}/final/` — **never retrains or refits anything at
  request time.**
- Predict page: accepts a raw CT scan (`.nii.gz`), a urinary-biomarker row, or both; shows
  Grad-CAM (imaging) and SHAP (clinical) explanations alongside the score.
- Risk bands (UI convention, not a clinical cutoff): Low <0.30, Moderate 0.30–0.70, High
  ≥0.70.
- Built 2026-07-20 to 2026-07-21, ~2,480 new lines across 11 files.

---

## 6. Anticipated "Why" Questions — Full Q&A Bank

### General / architecture

**Q: Why two completely separate models instead of one joint multimodal model?**
A: No dataset exists with both CT scans and this urinary biomarker panel for the same
patients. Building a joint model would require either (a) fabricating paired training data,
which would be scientifically dishonest, or (b) training on two unrelated patient
populations as if they were one, which would silently conflate two different case-mix
distributions. Two independent branches + an explicit, disclosed fusion rule is the honest
architecture given this data constraint.

**Q: Why is the fusion weight fixed (0.4/0.6) instead of learned?**
A: There's no paired ground truth to fit weights against — fitting anything here would mean
inventing a training signal that doesn't exist. The weights are instead a stated, defensible
judgment call: imaging is discounted because of its disclosed confound-check finding, not
discounted further because its calibration is otherwise strong.

**Q: Isn't a fixed rule "less rigorous" than a trained fusion layer?**
A: Given the data constraint, a trained fusion layer would actually be *less* rigorous — it
would require fabricating pairings or fitting on a proxy that doesn't reflect the real joint
distribution. A disclosed, reasoned fixed rule is the more honest choice, not a shortcut.

**Q: Why calibrate both branches (Platt scaling) at all?**
A: Raw classifier scores (XGBoost's raw probability, the sigmoid output of the imaging
model) aren't guaranteed to reflect true probabilities — they can be systematically over-
or under-confident. Calibration matters doubly here because the fusion rule averages two
probabilities directly; averaging two miscalibrated scores would compound the distortion.

**Q: Why Platt scaling instead of isotonic regression, in both branches?**
A: Isotonic regression is flexible but needs thousands of samples to avoid staircase
overfitting. Both branches' *effective* sample size (unique patients, not repeated-CV rows)
is small — 590 for clinical, 361 for imaging — favoring the simpler, more stable parametric
option in both cases.

**Q: Why fit the calibrator on out-of-fold predictions instead of the final model's own
output?**
A: The final model is fit on all data with no held-out set — scoring it on its own training
data and calibrating against that would be circular (the model has already memorized every
label it would be "predicting"). OOF predictions are the model's performance on data it
never trained on for that fold, which is the only honest signal available to calibrate
against.

### Tabular / clinical branch

**Q: Why XGBoost over Logistic Regression or Random Forest?**
A: Because sensitivity/recall — the clinically prioritized metric, since a missed PDAC case
is far costlier than a false alarm — is best for XGBoost in 2 of the 3 validation schemes
(the main repeated-CV estimate and the strictest leave-one-site-out test), while its AUC is
statistically indistinguishable from Random Forest's in those same two schemes. Random
Forest only wins in the single, non-repeated cohort-out split — the weakest of the three
estimates.

**Q: Why not just pick the model with the highest AUC?**
A: AUC treats false positives and false negatives symmetrically. In a cancer-screening
context, missing a real case (false negative) is far more clinically costly than a false
alarm — so recall/sensitivity is prioritized deliberately, and this decoupled from
AUC-maximization in this project (XGBoost's AUC lead over RF is negligible; its recall lead
is not).

**Q: Why use three different CV schemes instead of just one?**
A: Each answers a different generalization question. Repeated 5×20 stratified CV gives the
statistically tightest average-case estimate. Cohort-out simulates deploying on a genuinely
different patient batch. Leave-one-site-out simulates deploying at a hospital/site the model
has never seen — the strictest real-world generalization test, since biomarker assays can
have real batch/site effects. A model that only looks good on one of these would be a
red flag, not a win.

**Q: Why 20 CV repeats instead of 10?**
A: More repeats narrows the noise band around the same underlying number — it cannot make a
model "better," only give a tighter, more defensible confidence interval. It was raised from
10→20 specifically because it was nearly free (590 patients, fast models) and every number
moved by ≤0.001 between the two — confirming it wasn't an attempt to search for a flattering
result.

**Q: Why MICE over KNN for the missing CA19-9 imputation?**
A: MICE won on all 4 tracked metrics (AUC, PR-AUC, early-stage recall, accuracy) in both
multi-fold schemes (repeated CV and leave-one-site-out), with early-recall gaps of 8–10
percentage points — the metric that matters most for a screening tool. KNN only wins on the
single cohort-out split, and even there it's a near-tie on PR-AUC.

**Q: Why is the imputer refit inside every CV fold instead of once on the whole dataset?**
A: Fitting the imputer on the full dataset first (including test-fold rows) would leak
information from the "unseen" patients into how their own missing values get filled — the
same information-leakage principle as fitting a `StandardScaler` on train+test together.
Every fold gets its own freshly-fit imputer, trained only on that fold's training rows.

**Q: Why not just drop patients with missing CA19-9 instead of imputing?**
A: 240/590 (40.7%) of patients are missing this one value — dropping them would discard
40% of the dataset for a single feature, a far worse tradeoff than a validated, fold-safe
imputation strategy that was shown to actually improve downstream performance.

**Q: Why drop REG1A but keep imputing CA19-9?**
A: REG1A's missingness is structural and total for an entire cohort (100% missing in
Cohort2) — there's no ground truth anywhere to validate an imputation against for that
cohort. CA19-9's missingness is spread across both cohorts with real observed values to
learn the imputation relationship from (350/590 measured) — a fundamentally different
missingness pattern that supports imputation.

**Q: Why LYVE1 as the top SHAP feature instead of the clinically familiar CA19-9?**
A: This is an empirical finding of the fitted model, not an assumption or design choice —
worth flagging explicitly since CA19-9 is the biomarker a clinical audience will expect to
dominate. It's also scale-dependent: LYVE1 leads in log-odds space, but CA19-9 edges ahead
in probability space (Spearman ρ=0.9643 between the two rankings, with this pair being the
one swap). Framed honestly: "which feature matters most" depends on which scale you ask
in, and that ambiguity should be part of the answer, not hidden.

**Q: Doesn't 41% of CA19-9 being imputed undermine its SHAP ranking?**
A: Partially, yes — and that's disclosed rather than glossed over. CA19-9's global
importance is a mix of genuine biomarker signal (from the 350 measured rows) and
imputer-model behavior (from the 240 imputed rows, which are themselves derived from the
other correlated features). A specific example patient (row 576) shows an imputed CA19-9
value as the single largest driver of a PDAC classification — reading that as "measured
CA19-9 drove this" would be incorrect.

**Q: Why TreeExplainer instead of a model-agnostic method like KernelExplainer or LIME?**
A: XGBoost is a tree ensemble, and TreeExplainer computes *exact* Shapley values for tree
models via a closed-form algorithm that walks the trees directly. KernelExplainer/LIME are
approximation methods designed for models with no closed-form solution — using one here
would be strictly worse (slower and approximate) for no benefit, since the exact method is
available and free.

**Q: Why explain the model in log-odds space instead of probability, if probability is
what a clinician reads?**
A: SHAP's additivity guarantee only holds in the space the trees directly produce (raw
margin/log-odds) — the calibrator is a separate, later squashing function. Explaining in
probability space would require an approximate method and lose the exact
reconstruction guarantee. The tradeoff (numbers aren't directly "percentage points of
risk") is disclosed, and a secondary probability-space check was run to confirm how much
the ranking actually changes (not much — see above).

### Imaging / CT branch

**Q: Why ResNet50-U-Net over the baseline CNN, given the baseline has better precision and
specificity?**
A: Recall/sensitivity is the clinically load-bearing metric for cancer detection — a missed
cancer slice is far costlier than a false alarm. resnet50_unet's recall is both higher
(98.3% vs. 88.5%) and dramatically more *stable* across folds (95.9–99.7% band vs.
baseline's 68.9–99.7% swings). The instability itself is disqualifying for a clinical
pipeline, independent of the mean numbers — a model whose operating point swings wildly
depending on which patients are held out can't be trusted to generalize predictably.
Additionally, only resnet50_unet produces the segmentation mask the project's derived
risk-score structurally depends on.

**Q: Why does the precision/specificity gap favor the baseline model — doesn't that matter?**
A: It's a real, honestly-stated tradeoff (baseline is more conservative, resnet50_unet more
sensitive), not dismissed. But given the clinical cost asymmetry (missed cancer >> false
alarm) and baseline's instability, the tradeoff favors resnet50_unet on balance.

**Q: Why move from 3-fold to 5-fold CV partway through the project?**
A: The 3-fold cap was a schedule constraint from the local 4GB-GPU era (each additional
fold cost real wall-clock time that mattered on a slow laptop GPU). Once training moved to
a rented cloud GPU, the 2 extra folds cost ~27 minutes and under $1 — a cost/benefit
calculation that had genuinely changed, not a decision to keep re-running until a better
number appeared. 5-fold gives every one of the 361 patients exactly one out-of-fold
prediction, versus only 60.7% coverage at 3 folds.

**Q: Why 2D slices instead of full 3D volumes?**
A: A direct hardware constraint — the local development GPU (RTX 3050, 4GB VRAM) couldn't
support 3D volumetric training at any reasonable batch size. This is disclosed as a real
constraint, not framed as an unconditionally better design choice.

**Q: Why did you specifically go looking for a confound, instead of just reporting the
0.993 AUC / 0.963 recall?**
A: Because `dataset` (MSD vs. NIH) and `class` (cancer vs. healthy) are perfectly correlated
in this data by construction — MSD is 100% cancer, NIH is 100% healthy. That means a model
could reach these exact same headline numbers by learning to recognize which scanner
produced the image rather than real pathology, and no standard metric can tell the two
explanations apart. Reporting the numbers without checking for this would risk presenting a
shortcut as clinical competence.

**Q: Why trust the occlusion test (Round 4) over the gradient-based attribution methods
(Rounds 1–3), when the attribution evidence initially looked more reassuring?**
A: The attribution methods (Grad-CAM and its variants, even gradient-free EigenCAM) all
still only *infer* importance from gradients or activations — none of them intervene on the
actual input. Occlusion physically blanks a region and re-runs the model, directly measuring
the causal effect on the prediction. Causal evidence from manipulating the input outranks
correlational evidence from gradients, which is a general principle, not specific to this
project — that's why the more alarming, causally-grounded result was trusted over the more
reassuring, purely-inferential one.

**Q: Why did the first occlusion run find nothing, and how was that resolved?**
A: Because the model's predictions on correctly-classified slices are already saturated
(baseline P(cancer)≈0.9991) — the sigmoid is flat there, so no local occlusion can move a
*probability* much regardless of true importance. Switching to logit space (unbounded, never
saturates) revealed the real, statistically significant effect that probability space was
masking. This was a genuine methodological correction made mid-analysis, disclosed as such.

**Q: The padding shortcut got fixed — why didn't the tumour-under-reliance problem get
fixed too?**
A: Two different, not-yet-fully-understood mechanisms. The padding fix (random-resized-crop
augmentation) specifically targeted padding as a stable, exploitable per-slice fingerprint —
varying its position/amount per epoch removed that specific cue, and this was verified
directly (2.03x→1.08x, no longer distinguishable from control). Tumour under-reliance is a
different problem: making other regions unreliable (the mask-preserving-erase attempt in
Round 6) did not make the model specifically prioritize the tumour — it may instead now rely
on some more diffuse combination of surrounding tissue. Two possible, not-yet-distinguished
explanations remain open: a subtler residual shortcut, or the possibility that raw tumour
pixels alone are genuinely a weaker standalone signal than the full anatomical context
(similar to how radiologists don't read tumours in complete isolation from surrounding
anatomy either).

**Q: Why was Round 6 (mask-preserving erase) reverted instead of kept, since it didn't
clearly hurt anything?**
A: It didn't demonstrate any improvement on the problem it targeted (tumour reliance
unchanged, 0.58x vs. 0.62x, both still significant), and its effect on the already-fixed
padding shortcut was genuinely ambiguous — the point estimate rose to 2.86x, well above the
original problem's 2.03x, even though it didn't reach statistical significance at that
sample size. Given no upside and a real, un-ruled-out risk of quietly reintroducing a
previously-fixed problem, reverting to the verified Round 5 state was the more defensible
choice. The code and results weren't discarded — fully archived for a future, better-targeted
attempt (e.g. attention-supervision loss, the next-ranked option).

**Q: Is the imaging model unreliable, then? Should it even be used?**
A: The detection metrics (ROC-AUC 0.993, recall 0.963) are real and reproducible — the
model does discriminate cancer from healthy scans reliably within this dataset. What isn't
yet confirmed is *why* — specifically, that it's doing so via genuine tumour-specific
pixel-level reasoning rather than some remaining diffuse, non-anatomical cue. This is a
disclosed limitation carried into every downstream artifact (model card, fusion evaluation,
dashboard), not a reason to discard the model, but a reason to present its numbers with the
caveat attached every time.

**Q: Why is imaging weighted below tabular in the fusion rule, given imaging's raw metrics
look stronger?**
A: Directly because of the confound-check finding above — imaging's high confidence isn't
yet confirmed to reflect the reasoning it appears to. The raw numbers alone would justify
weighting it *higher*; the disclosed limitation is the specific, stated reason it's
discounted instead.

**Q: Why does averaging slice scores (mean aggregation) win, and is that itself informative?**
A: It wins empirically on F1 and Brier score against 6 alternative rules (median, max,
top-k, percentile). But *why* it wins is itself a second piece of evidence for the same
confound: per-slice cancer scores barely vary within a patient (std≈0.05), even on slices
that don't show the tumour at all — consistent with the model reading a global,
volume-level "cancer-ness" cue rather than localizing to tumour-bearing slices. If the model
genuinely localized tumours, `max`/top-k aggregation would outperform `mean`, and it
doesn't.

**Q: Why use ImageNet-pretrained weights for a medical CT model — isn't that a mismatch
(natural images vs. medical images)?**
A: Pretrained low/mid-level visual features (edges, textures, gradients) still transfer
usefully even across a domain gap, and starting from them is standard practice given a
comparatively small training set (361 patients) relative to what training a 71M-parameter
encoder from scratch would need. It's also empirically supported here: 2 of 5 folds reached
their best validation AUC at epoch 0 (`best_epoch=0`), meaning the pretrained features
needed essentially no fine-tuning to be useful on this task before the model started
overfitting — direct evidence the pretraining was doing real work, not just imported
irrelevant baggage.

**Q: Why a fixed epoch count (6) for the final all-data fit instead of early stopping?**
A: Early stopping requires a held-out validation set to monitor, and the final fit is
deliberately trained on *all* 361 patients with no held-out data (same "use everything, the
validation work is already done" philosophy as the clinical branch's final fit). The epoch
count was instead derived transparently from the 5-fold CV run's own per-fold best-epoch
values (mean 5.2 → rounded to 6), not picked arbitrarily.

### Data / methodology in general

**Q: Why is nothing in this project tuned by grid search / hyperparameter optimization?**
A: Hyperparameters were fixed by the project's governing spec (e.g., XGBoost's
`n_estimators=100, max_depth=3`) rather than independently tuned per model — explicitly
flagged as a real limitation in the tabular model comparison, not hidden. The stated
implication: if either model's hyperparameters were tuned harder, the gap between
candidates could narrow or shift, and the comparison would be worth re-deriving if the
predictor set or hyperparameters change materially.

**Q: How do you know the reported cross-validation numbers aren't leaking information?**
A: Every fold-safety rule is enforced deliberately and checked: imputers/scalers refit
fresh inside every training fold only; patient-level (not slice-level) splitting for
imaging, so no patient's anatomy appears in both train and test; verified explicitly that
every imaging patient's slices sit entirely within one CV fold and that every patient has
one consistent label across all its slices, before any patient-level aggregation is trusted.

**Q: Why report NaN instead of 0 when a metric is undefined (e.g. recall on an all-negative
test fold)?**
A: Recall is mathematically undefined (0/0), not 0, when a fold has zero true positive
cases to detect (e.g. the UCL site, 100% Benign, 0 PDAC patients) — silently reporting 0.0
would misrepresent an undefined quantity as a real, poor score. This NaN-not-a-manufactured-
number principle is applied consistently everywhere in the project (AUC, PR-AUC, recall,
early-stage recall) rather than only in cases that happen to be convenient.

**Q: Why can't you report one single joint accuracy number for the whole fused system?**
A: Because no patient in this project has both a CT scan and a urine sample — there is no
ground truth to score a fused prediction against. This is stated as a hard, structural
limitation of the available data, not a shortcut avoided or a metric withheld. Reporting a
fabricated joint number, even a plausible-looking one, would misrepresent what's actually
been validated.

---

## 7. Key Numbers Cheat-Sheet (for slide callouts)

| Metric | Value |
|---|---|
| **Tabular — XGBoost, repeated 5×20 CV** | AUC 0.9077, Recall 0.7433, F1 0.7573 |
| **Tabular — XGBoost, leave-one-site-out** | AUC 0.8195, Recall 0.7783 |
| **Tabular — imputer chosen** | MICE (IterativeImputer/BayesianRidge) over KNN |
| **Tabular — SHAP top feature (log-odds)** | LYVE1 (mean\|SHAP\|=1.669), CA19-9 close 2nd (1.293) |
| **Tabular — SHAP additivity check** | PASS, max error 4.77e-06 |
| **Imaging — resnet50_unet, 5-fold CV** | ROC-AUC 0.9934, Recall 0.9628, Precision 0.9920 |
| **Imaging — baseline_cnn, 5-fold CV** | ROC-AUC 0.9840, Recall 0.9073 |
| **Imaging — confound check (Round 4, causal)** | Tumour occlusion 0.47x control (p=0.0075) — under-relies on tumour; Padding occlusion 2.03x control (p=0.044) — over-relies on padding |
| **Imaging — confound check (Round 5, post-fix)** | Padding fixed: 1.08x control (p=0.917). Tumour under-reliance still open: 0.62x (p=0.0065) |
| **Fusion rule** | `0.4 × imaging + 0.6 × tabular`, fixed/hand-set, not fitted |
| **Fusion aggregation** | `mean` over per-slice imaging probabilities (F1 0.9893, Brier 0.0142) |
| **Fusion — joint benchmark** | None exists — no paired patients in the data |
| **Calibration (Brier, lower=better)** | Tabular 0.1189, Imaging 0.0282 (both post-calibration) |

---

*Compiled 2026-08-12 from the project's own executed notebooks and documentation. Every
number above traces back to a specific source doc listed at the top of this file — cross-
reference there if a specific slide needs the deeper methodology behind any figure.*
