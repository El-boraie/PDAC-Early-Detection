# FYP Demo Video — Script & Shot List (v2, up to 7:00)

**Purpose:** Demonstrate the end-to-end pipeline for the PDAC multimodal
detection system, per the CSDA brief: data importing → preprocessing → model
building → evaluations → model deployment.

**Target runtime:** ~7:00 maximum. Word counts assume ~140 words/min spoken —
read each block aloud once before recording to check it lands near its target;
trim live if you're running long (cut guidance at the bottom).

**Format decision (confirmed):** Sections 1–4 (data, preprocessing, model
building, evaluation) are shown by scrolling/running the **actual notebooks**
— real cells, real already-executed outputs — not static cropped images.
Section 5 (deployment) is the one live, non-notebook segment: the running
Streamlit dashboard, since that's the actual deployed artifact and the brief
explicitly asks for "model deployment."

**Recording mechanics — decide once, before you start:**
- Either screen-record each notebook open in Jupyter/VS Code with outputs
  already executed (fastest, no risk of a cell failing on camera), **or**
  re-run cells live if you want to show execution — if you do this, run the
  whole notebook once, off-camera, immediately beforehand so kernel state is
  warm and nothing hangs on camera.
- Have every file below already open in its own tab, in the order listed, so
  you switch tabs instead of using File→Open on camera.
- Maximize the window / hide the file tree so output plots are as large as
  possible on screen.

---

## Pre-flight checklist — open these, in this order

1. `notebooks/CT EDA.ipynb`
2. `notebooks/Tabular EDA.ipynb`
3. `docs/CT_Preprocessing_documentation.md` and `src/imaging/preprocessing.py`
4. `src/imaging/models.py` and `src/imaging/train_segmentation_detection.ipynb`
5. `src/clinical/clinical_model_comparison.ipynb` and
   `src/clinical/clinical_final_fit.ipynb`
6. `src/imaging/imaging_confound_check.ipynb` and
   `src/imaging/imaging_evaluation.ipynb`
7. `src/clinical/clinical_shap.ipynb`
8. `src/fusion/fusion_evaluation.ipynb` and `src/fusion/fusion.ipynb`
9. The dashboard running locally (`streamlit run dashboard/app.py`) — test the
   full Register → Predict → PDF click-path once, off-camera, with one real
   cached `.nii.gz` file and one biomarker row ready to paste in.

---

## 0. Cold open (0:00–0:15) — ~35 words

**Screen:** title card, or the dashboard's About page title/header.

> "This is my Final Year Project: an interpretable, multimodal machine
> learning system for early pancreatic cancer detection, fusing CT imaging
> and urinary biomarker data into a single risk score. I'll walk through the
> full pipeline — from raw data to a deployed dashboard."

---

## 1. Data importing (0:15–1:05) — ~50s, ~117 words

**Open `CT EDA.ipynb`.**

**Show, in this order:**
- **Section B — Dataset Structure** (`B1. Paths`, `B2. Path Verification`):
  scroll to the executed output showing the resolved dataset paths.
- **Section D — Class Labels and Balance** (`D1. Patient and Slice-Level
  Balance`): the printed/plotted patient and slice counts.
- **Section L — NIH additions** (`L4. Merged Patient-Level Class Balance`,
  `L5. Merged Slice-Level Balance`): the combined MSD+NIH totals.

> "Two datasets feed the imaging branch. MSD Pancreas contributes 281 cancer
> patients; NIH Pancreas-CT contributes 80 healthy patients — added
> specifically so the model sees negative cases at all, not only tumours.
> Combined, that's 361 patients and just over ninety thousand processed CT
> slices, verified here in the EDA notebook's own class-balance output."

**Switch to `Tabular EDA.ipynb`.**

**Show:** `A2. Dataset Overview` (head/shape of the raw dataframe) and
`E1. Target Class Distribution`.

> "On the clinical side: the Debernardi et al. urinary biomarker dataset, 590
> patients, seven usable features including CA19-9, the standard clinical
> marker for this cancer."

---

## 2. Preprocessing (1:05–1:50) — ~45s, ~105 words

**Stay in `Tabular EDA.ipynb`.**

**Show:** `D1. Missing Count/% Per Column` and `F1. Skewness — All
Biomarkers` (this is the cell that found CA19-9 skewness = 8.02, not the
10.37 the original plan assumed — a real, corrected number).

> "The EDA is what drives every preprocessing decision. CA19-9 is missing in
> about 41% of patients, and its distribution is heavily right-skewed — both
> of which shaped the imputation choice."

**Switch to `docs/CT_Preprocessing_documentation.md`**, scrolled to the
pipeline-steps section, then briefly to `src/imaging/preprocessing.py`.

> "For CT scans, the fixed pipeline is: reorient to a standard axis
> convention, resample to 1-millimetre isotropic spacing, clip Hounsfield
> units to the soft-tissue window, normalize, and center-crop to a fixed 320
> by 320 box. This exact pipeline was later re-implemented as an importable
> module for the dashboard, and verified byte-for-byte against already-cached
> patients before being trusted for live uploads."

---

## 3. Model building — imaging branch (1:50–2:50) — ~60s, ~140 words

**Open `src/imaging/models.py`**, scrolled to the `ResNet50UNet` class
definition.

> "For imaging, I built a 2D U-Net with a pretrained ResNet-50 encoder — 2D,
> not 3D, because of the original hardware's 4 gigabytes of VRAM — with two
> output heads: segmentation, masked to slices that actually have a tumour
> annotation, and detection, cancer-versus-healthy, trained on every slice
> from both datasets."

**Switch to `train_segmentation_detection.ipynb`.**

**Show:** `## Models` section (the two candidate model definitions side by
side), then `## Full Run (gated by RUN_FULL_TRAINING)` scrolled to its
executed training-log output (epoch/loss lines).

> "I trained this against a simpler baseline CNN using genuine 5-fold
> cross-validation, so all 361 patients get exactly one out-of-fold
> prediction each — this is the actual training run, executed on a rented GPU
> after local hardware measured as too slow for the full schedule."

---

## 4. Model building — clinical branch (2:50–3:40) — ~50s, ~117 words

**Open `clinical_model_comparison.ipynb`.**

**Show:** `## Three Candidate Models` (XGBoost, Logistic Regression, Random
Forest defined together), then `## Comparison Table — All Metrics, All
(Model × Scheme) Cells` scrolled to its executed output table, then
`## Sensitivity (Recall) — Dedicated Section`.

> "For the clinical branch, I compared XGBoost against regularized logistic
> regression and random forest, across three different cross-validation
> schemes, prioritising recall — sensitivity — since missing a real cancer
> case is the clinically worse error. XGBoost won two of the three schemes on
> recall; random forest won the third, and that's disclosed honestly rather
> than hidden."

**Switch to `clinical_final_fit.ipynb`.**

**Show:** `## All-Data Fit — Imputer + Model, No Folds` and `## Save to
checkpoints/clinical/final/`.

> "The winning model is then refit on all 590 patients — no held-out folds —
> and saved as the deployable artifact."

---

## 5. Evaluation — imaging confound check (3:40–4:40) — ~60s, ~140 words

**Open `imaging_confound_check.ipynb`.**

**Show:** `## Visual Panels` (the Grad-CAM overlay images), then
`## Occlusion-Based Sensitivity Analysis` scrolled to its printed statistics,
then `## Paired Comparison and Final Verdict`.

> "Evaluation went beyond accuracy. In this dataset, every cancer patient
> comes from one source, MSD, and every healthy patient from another, NIH —
> so high accuracy alone can't prove the model is looking at the tumour
> itself rather than which scanner produced the image. I ran a direct causal
> test: physically blanking the real tumour region versus blanking an
> ordinary patch of tissue. The result was mixed, and I'm showing it exactly
> as found: the model relied significantly *more* on a synthetic
> zero-padding artifact than on real tissue — a real shortcut, which I then
> fixed and re-verified — but it also relied significantly *less* on the
> actual tumour than on ordinary tissue, and that second issue remains only
> partially resolved after two independent fix attempts."

**Switch to `imaging_evaluation.ipynb`.**

**Show:** `## The Winning Model: resnet50_unet` and `## Confound Check —
Summary` — the model card's `known_limitations` field, so it's visible this
is written into the deployed artifact itself, not just a side document.

> "That finding is written directly into the promoted model's own model
> card, so it travels with the artifact wherever it's used."

---

## 6. Evaluation — clinical SHAP (4:40–5:25) — ~45s, ~105 words

**Open `clinical_shap.ipynb`.**

**Show:** `## Figures` scrolled to the global-importance bar chart and
beeswarm plot, then `### CA19-9 dependence plot -- imputed rows marked`, then
`### Three individual waterfalls` scrolled to the row-576 example.

> "For the clinical branch, SHAP explains which biomarkers drive each
> prediction. Interestingly, LYVE1 — a less familiar urinary marker —
> outranks the more clinically familiar CA19-9 in this fitted model. But
> that ranking needs a caveat: 41% of patients never had CA19-9 measured, so
> for those rows, SHAP is explaining the imputer's prediction, not a real lab
> value — visible here as the orange markers on the dependence plot, and in
> this worked example, where an imputed CA19-9 value is the single largest
> driver of a cancer classification."

---

## 7. Fusion (5:25–6:05) — ~40s, ~93 words

**Open `fusion_evaluation.ipynb`.**

**Show:** `## The One Real Pair: resnet50_unet vs. XGBoost` scrolled to the
side-by-side metrics table.

> "Because no patient has both a CT scan and a urine sample, there's no
> paired ground truth to compute a joint accuracy metric — so fusion compares
> each branch's own metrics side by side instead, honestly, rather than
> fabricating a number that doesn't exist."

**Switch to `fusion.ipynb`.**

**Show:** `## The Two Fixed Rules — Named Constants` (the `W_IMAGING = 0.4`,
`W_TABULAR = 0.6` cell) and `## Measurement: Slice-Level vs. Volume-Level
Aggregation` scrolled to its comparison table.

> "The two branches are combined with a fixed 0.4/0.6 weighted average —
> imaging weighted down deliberately, because of the confound-check finding
> just shown — and a measured rule for aggregating a full CT volume's
> per-slice scores into one patient-level score. Both are documented judgment
> calls, not fitted parameters, since there's nothing to fit them against."

---

## 8. Deployment — live dashboard walkthrough (6:05–6:50) — ~45s narration + live demo

**Screen:** live screen recording of the running Streamlit app
(`streamlit run dashboard/app.py`).

**Show, in this order, clicking through live:**
1. **Register page** — fill in a demo patient (name, DOB, sex), submit.
2. **Predict page** — upload the real `.nii.gz` test file; paste in the
   biomarker row; click predict; wait for it to resolve.
3. **Result view** — the fused risk score, the Grad-CAM overlay on the CT
   slice, and the SHAP chart for the biomarker contributions, side by side.
4. **Reports page** (or a generated PDF opening) — briefly.

> "All of this is deployed in a Streamlit dashboard. A patient is registered
> once, then on Predict I upload a real CT scan — reprocessed live through
> the exact pipeline shown earlier, not a bundled sample — and enter a
> biomarker reading. The system runs both branches, combines them into one
> fused score using the same 0.4/0.6 rule, and shows the Grad-CAM overlay and
> SHAP chart together. Every case is saved and exportable as a PDF report."

---

## 9. Close (6:50–7:00) — ~35 words

**Screen:** Analytics page, scrolled to the disclosure section, or the About
page.

> "The dashboard also includes an analytics view across every recorded case,
> and one persistent, honest disclosure of the model's known limitations —
> because a defensible result has to be reported as it actually is, not as
> it looks best."

---

## Timing summary

| # | Section | Length | Cumulative |
|---|---|---|---|
| 0 | Cold open | 0:15 | 0:15 |
| 1 | Data importing (CT EDA + Tabular EDA) | 0:50 | 1:05 |
| 2 | Preprocessing (EDA → pipeline decisions) | 0:45 | 1:50 |
| 3 | Model building — imaging | 1:00 | 2:50 |
| 4 | Model building — clinical | 0:50 | 3:40 |
| 5 | Evaluation — imaging confound check | 1:00 | 4:40 |
| 6 | Evaluation — clinical SHAP | 0:45 | 5:25 |
| 7 | Fusion | 0:40 | 6:05 |
| 8 | Deployment (live dashboard) | 0:45 | 6:50 |
| 9 | Close | 0:10 | 7:00 |

## If you need to cut for time

Cut in this order (least to most costly):
1. Section 1's NIH merged-balance sub-shot (L4/L5) — mention the combined
   total verbally without showing the cells.
2. Section 7 (Fusion) down to one sentence and one screen (the 0.4/0.6 cell
   only, skip the aggregation-measurement table).
3. Section 4's random-forest cohort-out caveat — mention "XGBoost won" only.

**Do not cut:** the occlusion-test result in Section 5 (it's the single most
defensible, distinctive piece of methodology in the project) or the live
Predict-and-explain moment in Section 8 (the one thing a reader can't get
from the written report).

## If you have slack time (running under 6:30)

Add one sentence in Section 3 on why 2D-not-3D and ResNet-50-not-scratch were
chosen (already in the handoff, Section 2 of `PDAC_FYP_Project_Handoff.md`),
or extend Section 6 with the log-odds-vs-probability-space ranking flip
(`## Robustness check` in `clinical_shap.ipynb`) — a genuine secondary
finding currently only mentioned in passing.
