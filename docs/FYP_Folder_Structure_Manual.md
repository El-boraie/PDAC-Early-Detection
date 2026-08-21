# PDAC FYP — Structure Manual

**Purpose:** a plain-language reference for what every file in the project does,
and its current build status. Written after the v3 folder migration and the
rollback described in the same session. Use this to orient yourself, review what
Claude Code has actually built, and know what's still ahead.

**Status key:** `DONE+VERIFIED` = built, run, numbers checked. `STUB` = file
exists with structure/imports but no working logic. `NOT YET CREATED` = doesn't
exist. `N/A` = nothing to build, it's a folder or a data file.

---

## data/

- **`data/raw/`** — the original downloads, never touched. `Task07_Pancreas/`
  (MSD CT scans), `NIH_Pancreas_CT/0024_pancreas_ct/` (NIH CT scans),
  `Debernardi et al 2020 data.csv` (urine biomarker table, 590 patients).
  `N/A` — this is source data, not something built.

- **`data/processed/images/{train,val,test}/`, `images/nih/`** — the CT scans
  after preprocessing (reoriented, resampled, HU-windowed, saved as `.npy`
  slices). Note: actual folder names are `train/val/test` for MSD (from the
  original holdout split) plus a separate `nih/` — not `msd/nih/` as the tree
  diagram showed; `manifest.csv`'s `dataset` column tells you which is which
  either way. `DONE+VERIFIED` — 90,693 slices, produced by `imaging.ipynb`.

- **`manifest.csv`** — the master index of every CT slice: which patient, which
  dataset (MSD/NIH), which class, which fold (0–4). Everything imaging touches
  reads this file. `DONE+VERIFIED`.

- **`manifest_msd_only_backup.csv`** — a safety copy of the manifest from before
  NIH was merged in, in case the merge ever needs undoing. `DONE`, not actively used.

- **`tabular_clean.csv`** — the urine biomarker table after fixed-rule cleaning
  only: labels mapped to readable names, structurally-missing fields
  sentinel-filled, `REG1A` dropped. **`plasma_CA19_9` is deliberately left blank**
  — imputing it is a *learned* step and has to happen inside cross-validation,
  never baked into this file. `DONE+VERIFIED` — 590 rows, produced by
  `tabular_clean.ipynb`.

---

## outputs/ (figures for the report — safe to commit to git)

- **`outputs/eda/ct/`** — 10 files (8 PNGs + 2 CSVs) from `CT EDA.ipynb`:
  slice-count distributions, HU histograms, lesion-size spread, etc.
  `DONE+VERIFIED`.
- **`outputs/eda/tabular/`** — 25 files from `Tabular EDA.ipynb`: biomarker
  distributions, skewness, missingness-by-cohort, correlation heatmap, etc.
  `DONE+VERIFIED`.
- **`outputs/qa/ct/`** — the two preprocessing sanity-check images
  (`qa_preprocessed_overlay.png`, `qa_nih_preprocessed.png`), moved here from
  `data/processed/` since they're QA figures, not pipeline inputs. `DONE`.
- **`outputs/qa/tabular/`** — reserved for tabular QA figures, currently empty.
  `N/A` for now.
- **`outputs/eval/{imaging,clinical,fusion}/`** — where Grad-CAM overlays, SHAP
  plots, ROC/PR curves, and reliability diagrams will land once each branch is
  actually evaluated. `NOT YET CREATED` (folders exist, empty).

## results/ (metrics tables for the report — safe to commit to git)

- **`results/clinical/imputer_benchmark.csv`** — the KNN-vs-MICE comparison
  numbers (AUC, PR-AUC, early recall, accuracy across all three CV schemes).
  This is what proved MICE wins. `DONE+VERIFIED`.
- **`results/clinical/model_comparison.csv`, `oof_predictions.csv`** — where
  the XGBoost/LogReg/RF comparison and its out-of-fold predictions will be
  written. `NOT YET CREATED` (rolled back — see the prompt below).
- **`results/imaging/`, `results/fusion/`** — same idea for the other two
  branches. `NOT YET CREATED`.

---

## src/clinical/ (tabular branch)

- **`tabular_clean.ipynb`** — loads the raw CSV, maps labels, sentinel-fills
  structurally-missing fields, drops `REG1A`, writes `tabular_clean.csv`.
  **This is your "preprocessing" step.** `DONE+VERIFIED`.

- **`clinical_imputer_benchmark.ipynb`** — loads `tabular_clean.csv`, builds
  two candidate imputers for the missing `plasma_CA19_9` values (KNN and MICE),
  runs both through the same CV folds, and compares them. **This is your
  "imputation" step — the last thing you'd actually approved.** Result: MICE
  wins clearly (repeated CV AUC 0.9081±0.0229 vs. 0.891±0.027; the same run
  also happens to reproduce the pre-split `clinical.ipynb` numbers exactly,
  which is why no separate verification notebook was strictly needed).
  `DONE+VERIFIED`.

- **`clinical_model_comparison.ipynb`** — would compare XGBoost vs. logistic
  regression vs. Random Forest (using the now-settled MICE imputer) to justify
  XGBoost as the chosen model. **This is past where you'd approved work to
  stop — rolled back, to be rebuilt when you're ready to start that stage
  together.** `NOT YET CREATED` (after rollback).

- **`clinical_final_fit.ipynb`** — would do the one-time final fit (imputer +
  winning model + calibrator) on all 590 patients, producing the actual
  deployable artifact. Depends entirely on the comparison step above.
  `NOT YET CREATED` (after rollback).

---

## src/imaging/ (CT branch)

- **`imaging.ipynb`** — the full CT preprocessing pipeline: reorient, resample
  to 1mm isotropic, HU-clip and normalize, save as `.npy`, build the manifest.
  Only change made during migration: two QA image save-paths redirected to
  `outputs/qa/ct/`. Everything else untouched. `DONE+VERIFIED`.

- **`train_segmentation_detection.ipynb`** — will train the two-head U-Net
  (segmentation + cancer detection) and a baseline CNN candidate, using the
  manifest's 5-fold split. Currently reads the manifest correctly and has a
  `FOLDS_TO_RUN = [0]` config point ready, but the actual training loop is a
  stub — nothing has been trained. `STUB`.

- **`imaging_evaluation.ipynb`** — will compare the two imaging candidates,
  pick a winner, fit its calibrator, and run the Grad-CAM/segmentation-overlap
  check (this checks whether the model is reading real pathology vs. just
  picking up on MSD/NIH being different scanners — a real methodological
  concern, not a formality). Structure exists, logic is a stub. `STUB`.

---

## src/fusion/

- **`fusion.ipynb`** — will load both branches' final models and calibrators
  and combine their scores with a fixed rule, for the dashboard. `STUB`
  (pre-existing empty file, not built yet).

- **`fusion_evaluation.ipynb`** — will compare different (imaging-model,
  tabular-model) pairs by reporting each branch's own metrics side-by-side —
  deliberately **no single joint score**, since no patient has both a CT scan
  and a urine sample to validate a combined prediction against. Structure
  exists, logic is a stub. `STUB`.

---

## src/utils/

- **`config.py`** — every file path the new notebooks need (raw data,
  processed data, outputs, results, checkpoints), in one place, plus a random
  seed and a helper to create missing folders. New notebooks import from here
  instead of hardcoding paths — so if the project ever moves off `C:\FYP\`,
  only this file changes. `DONE`.

- **`metrics.py`** — shared scoring code. `early_stage_recall` and `pr_auc` are
  real, working functions, lifted directly from the already-verified
  `clinical.ipynb` logic (needed to check the imputer benchmark's numbers).
  `dice_score`/`iou_score` are named but unimplemented placeholders — there's
  no segmentation model yet to test them against. `PARTIAL — clinical metrics
  done, imaging metrics stubbed`.

- **`utils.ipynb`** — a pre-existing empty notebook, not referenced anywhere.
  `N/A`.

---

## src/_archive/

- **`clinical.ipynb`** — your original, fully verified tabular notebook, moved
  here unmodified once its logic was confirmed to reproduce correctly in the
  split notebooks. Kept as provenance — not deleted, not imported by anything.
  If anyone ever asks "how were these numbers originally produced," this is
  the answer. `DONE — archived, read-only in spirit`.

---

## notebooks/ (EDA — exploration only)

- **`CT EDA.ipynb`** *(actual filename has a space, not an underscore — kept
  as-is, same principle as not renaming the raw CSV)* — exploratory look at the
  CT data: slice counts, HU distributions, lesion sizes. Figures now save to
  `outputs/eda/ct/`. `DONE+VERIFIED`.

- **`Tabular EDA.ipynb`** — exploratory look at the biomarker data: skewness,
  outliers, missingness patterns, correlations. Figures now save to
  `outputs/eda/tabular/`. `DONE+VERIFIED`.

---

## checkpoints/ (gitignored — trained artifacts only, nothing here yet)

- **`checkpoints/imaging/candidates/`, `checkpoints/imaging/final/`** — will
  hold the trained U-Net/CNN weights and the winner + its calibrator. Empty.
- **`checkpoints/clinical/final/`** — will hold the final-fit XGBoost model,
  the MICE imputer, and the calibrator. Empty.
- All `NOT YET CREATED` — nothing has been trained.

---

## dashboard/

- **`app.py`** — the Streamlit app: CT upload → segmentation + score; tabular
  upload → prediction; combined view → fusion score + explanations. Will only
  ever read from `checkpoints/*/final/`, never retrain. `NOT YET CREATED`.

---

## docs/

Every existing writeup, moved into one place: `CT_EDA_documentation.md`,
`CT_Preprocessing_documentation.md`, `Tabular_EDA_documentation.md`,
`Tabular_Preprocessing_documentation.md`, plus this migration file. **Six files
referenced by earlier planning docs
(`CLAUDE_CODE_BRIEF.md`, `PDAC_FYP_Folder_Structure_Proposal.md`,
`Tabular_EDA_Summary_and_Open_Issues.md`, `Tabular_Everything_Else.md`,
`Tabular_Preprocessing_plan.md`, `docs/chapters/`) don't actually exist on this
machine** — same "exists only on another machine/Drive" gap flagged once
already for the EDA notebooks. If you have these in Drive, they need to be
copied in manually; nothing in the code depends on them.

---

## Root files

- **`PROJECT_HANDOFF.md`** *(actual filename — not `PDAC_FYP_Project_Handoff.md`
  as some planning docs assumed)* — the running technical record of the whole
  project. Source of truth is the Drive copy.
- **`requirements.txt`, `requirements-lock.txt`, `.gitignore`, `SETUP.md`** —
  environment setup, unchanged.
