# PDAC FYP — Folder Structure Migration (v3, final)

**Purpose of this file:** instructions for Claude Code to restructure the existing
`C:\FYP\` project into the shape below, before model training begins on either
branch.

**This is v3 and supersedes everything before it** — `PDAC_FYP_Project_Handoff.md`
Section 5, `PDAC_FYP_Folder_Structure_Proposal.md`, and migration v1/v2. If a
partially-applied earlier shape is found on disk, migrate it to v3 rather than
layering on top.

**Student:** Aley Ahmed Nabil Elboraie (TP075961) — APD3F2511CS(DA)
**Deadline:** 22 July 2026

---

## 0. The two rules this whole structure follows

**Rule 1 — save vs. transient.** Anything that's a *fixed rule* gets saved to disk
once and reused. Anything that's a *learned guess* only exists transiently, inside
a fold, and gets thrown away — except for one final version at the very end.

This governs three learned steps in this project: the **CA19-9 imputer**
(Section 1), **per-branch calibration** (Section 6), and **model selection**
itself (Section 4).

**Rule 2 — the deployed thing must never depend on who wins.** Every path the
dashboard, fusion, or report reads from is fixed and generic. Which model, which
imputer, which encoder won is recorded *inside* metadata files, never encoded in
folder or file names. This is what makes the structure survive you changing your
mind (Section 8).

---

## 1. Locked decision: CA19-9 imputer

**Settled, not open.** `MICE_CA19_9Imputer` (`IterativeImputer(BayesianRidge())`)
beat KNN on both metrics that matter — repeated 5x10 CV: AUC 0.908+/-0.023 vs.
0.891+/-0.027, early recall 0.722+/-0.104 vs. 0.641+/-0.097; leave-one-site-out:
AUC 0.820+/-0.032 vs. 0.805+/-0.027, early recall 0.808+/-0.172 vs. 0.704+/-0.159.
KNN only edged ahead on cohort-out (a single split, noisiest estimate), and even
there PR-AUC was a near-tie.

KNN stays in `clinical_imputer_benchmark.ipynb` as a **report artifact only** —
never imported into the training path.

---

## 2. Locked decision: what "comparing models" means here

Comparison is **within** each branch, to justify the chosen model against
alternatives — never tabular-vs-imaging (different data, different tasks,
incomparable metrics).

- **Tabular:** XGBoost (chosen) vs. regularized logistic regression vs. Random
  Forest.
- **Imaging:** ResNet-50 U-Net (chosen) vs. a second baseline CNN.
- **Fusion:** different (imaging-model, tabular-model) **pairs**, not fusion
  *strategies*. **No joint fused metric exists.** No patient has both CT and
  urine data, so there is no ground truth to score a fused prediction against,
  regardless of which two models are paired. Fusion comparison reports each
  branch's own already-validated metrics side-by-side per pair — "best pair"
  means best-tabular-by-its-own-numbers + best-imaging-by-its-own-numbers.
  **State this explicitly in the report as a dataset limitation.** Do not present
  it as if a fused benchmark exists.

---

## 3. Full target tree

```
C:\FYP\
|-- data\
|   |-- raw\                                    # never touched, ever
|   |   |-- Task07_Pancreas\
|   |   |-- NIH_Pancreas_CT\0024_pancreas_ct\
|   |   `-- Debernardi et al 2020 data.csv      # original filename kept, spaces and all
|   |
|   `-- processed\                              # fixed-rule pipeline INPUTS only
|       |-- images\{msd,nih}\*.npy              #   no figures, no metrics, no models
|       |-- manifest.csv
|       |-- manifest_msd_only_backup.csv
|       `-- tabular_clean.csv                   # labels mapped, sentinel-filled,
|                                                #   REG1A dropped, CA19-9 left NaN
|
|-- outputs\                                     # FIGURES (.png) — for the report
|   |-- eda\
|   |   |-- ct\                                 # CT_EDA.ipynb figures
|   |   `-- tabular\                            # Tabular_EDA.ipynb figures
|   |-- qa\
|   |   |-- ct\                                 # qa_preprocessed_overlay.png,
|   |   |                                       #   qa_nih_preprocessed.png
|   |   |                                       #   -- MOVED from data/processed/
|   |   `-- tabular\                            # reserved
|   `-- eval\                                   # NEW in v3 — evaluation figures had
|       |-- imaging\                            #   NO HOME in v2. Grad-CAM overlays,
|       |                                       #   seg-overlap panels, ROC/PR curves,
|       |                                       #   reliability diagrams
|       |-- clinical\                           #   SHAP plots, ROC/PR, confusion
|       |                                       #   matrices, reliability diagrams
|       `-- fusion\                             #   side-by-side comparison figures
|
|-- results\                                     # METRICS (.csv) — for the report
|   |-- clinical\
|   |   |-- model_comparison.csv                # XGB/LogReg/RF x 3 CV schemes
|   |   |-- imputer_benchmark.csv               # KNN vs MICE (Section 1's numbers)
|   |   `-- oof_predictions.csv                 # out-of-fold preds -- feeds the
|   |                                           #   calibrator, see Section 6
|   |-- imaging\
|   |   |-- model_comparison.csv                # ResNet50-UNet vs baseline CNN
|   |   `-- oof_predictions.csv                 # per-fold held-out preds
|   `-- fusion\
|       `-- pair_comparison.csv                 # side-by-side, no joint column
|
|-- src\
|   |-- imaging\
|   |   |-- imaging.ipynb                       # preprocessing -- DONE, do not rewrite
|   |   |-- train_segmentation_detection.ipynb  # NEW -- trains each candidate per fold
|   |   `-- imaging_evaluation.ipynb            # NEW -- compares candidates, writes
|   |                                           #   results/ + outputs/eval/imaging/,
|   |                                           #   promotes winner to final/,
|   |                                           #   fits calibrator, runs Grad-CAM
|   |                                           #   confound check
|   |
|   |-- clinical\
|   |   |-- tabular_clean.ipynb                 # NEW -- fixed-rule cleaning only
|   |   |-- clinical_imputer_benchmark.ipynb    # NEW -- KNN vs MICE, report artifact
|   |   |-- clinical_model_comparison.ipynb     # NEW -- XGB/LogReg/RF, MICE only,
|   |   |                                       #   3-scheme CV, writes results/ +
|   |   |                                       #   oof_predictions.csv
|   |   `-- clinical_final_fit.ipynb            # NEW -- one-time all-data fit +
|   |                                           #   calibrator -> checkpoints/
|   |
|   |-- fusion\
|   |   |-- fusion.ipynb                        # loads final models + calibrators,
|   |   |                                       #   applies FIXED combination rule
|   |   `-- fusion_evaluation.ipynb             # NEW -- pair comparison table
|   |
|   |-- utils\
|   |   |-- config.py                           # all paths live here, nowhere else
|   |   `-- metrics.py
|   |
|   `-- _archive\                               # NEW in v3 -- retired-but-not-deleted
|       `-- clinical.ipynb                      #   the original verified notebook,
|                                               #   moved here after its logic is
|                                               #   split out. See Section 5.
|
|-- notebooks\                                   # exploration only -- EDA
|   |-- CT_EDA.ipynb                            # -> outputs/eda/ct/
|   `-- Tabular_EDA.ipynb                       # -> outputs/eda/tabular/
|
|-- checkpoints\                                 # gitignored -- trained things only
|   |-- imaging\
|   |   |-- candidates\                         # one SUBFOLDER per candidate, because
|   |   |   |-- resnet50_unet\                  #   k=5 folds means 5 .pt files each,
|   |   |   |   |-- fold_0.pt ... fold_4.pt     #   not one (v2 got this wrong)
|   |   |   |   `-- model_card.json
|   |   |   `-- baseline_cnn\
|   |   |       |-- fold_0.pt ... fold_4.pt
|   |   |       `-- model_card.json
|   |   `-- final\                              # GENERIC NAMES -- see Rule 2
|   |       |-- model.pt                        #   whichever candidate won
|   |       |-- calibrator.pkl
|   |       `-- model_card.json                 #   records WHICH model this is
|   `-- clinical\
|       `-- final\                              # no candidates/ -- see Section 4
|           |-- model.pkl                       #   whichever candidate won
|           |-- ca19_9_imputer.pkl              #   MICE, final-fit only
|           |-- calibrator.pkl
|           `-- model_card.json
|
|-- dashboard\
|   `-- app.py                                   # loads checkpoints/*/final/ ONLY.
|                                                #   Never retrains. Never reads data/.
|                                                #   Never needs editing when a
|                                                #   different model wins.
|
|-- docs\
|   |-- FYP_Folder_Structure_Migration.md        # THIS FILE -- the current structure
|   |-- CT_EDA_documentation.md
|   |-- CT_Preprocessing_documentation.md
|   |-- Tabular_EDA_documentation.md
|   |-- Tabular_EDA_Summary_and_Open_Issues.md
|   |-- Tabular_Everything_Else.md
|   |-- Tabular_Preprocessing_plan.md
|   |-- Tabular_Preprocessing_documentation.md
|   |-- PDAC_FYP_Folder_Structure_Proposal.md    # superseded, kept for history
|   `-- chapters\                                # IR chapter drafts (.md -> Word)
|
|-- CLAUDE_CODE_BRIEF.md
|-- PDAC_FYP_Project_Handoff.md                  # root copy; Drive is source of truth
|-- SETUP.md
|-- requirements.txt
|-- requirements-lock.txt
`-- .gitignore
```

---

## 4. Why `checkpoints/clinical/` has no `candidates/` subfolder

**Intentional — do not "fix" this.** XGBoost/LogReg/RF fit in seconds, not
GPU-hours. Inside cross-validation each candidate is fit and discarded per fold
anyway (same fold discipline as the CA19-9 imputer). Nothing is worth persisting
per candidate except its *metrics* — that's `results/clinical/model_comparison.csv`.

Imaging is the opposite: training is GPU-expensive on a 4GB card, so re-training
a candidate just to re-evaluate it is wasteful. Hence `checkpoints/imaging/candidates/`,
with one subfolder per candidate holding its five per-fold checkpoints.

---

## 5. What happens to the existing `clinical.ipynb`

**This was undefined in v2 and must not be left to interpretation.**

`src/clinical/clinical.ipynb` currently exists, is executed, and produced the
verified numbers in Section 1. Its logic gets split three ways (cleaning ->
`tabular_clean.ipynb`, imputer benchmark -> `clinical_imputer_benchmark.ipynb`,
CV harness -> `clinical_model_comparison.ipynb`).

**Once the split is complete and the numbers are re-verified, move the original to
`src/_archive/clinical.ipynb`. Do not delete it.** It is the only executed record
of the verified results until the new notebooks have been run successfully. It is
not imported by anything and is not part of the pipeline — it exists purely as a
fallback and as provenance if an examiner asks how the numbers were originally
produced.

Delete it only after the new notebooks have run clean and matched Section 1.

---

## 6. Calibration, and where the out-of-fold predictions come from

The architecture combines **calibrated** probability scores at fusion. A
calibrator (Platt scaling or isotonic regression) is itself fit on data — so it
carries the same leakage risk as the CA19-9 imputer, and must be fit on
predictions the model has *not* been trained on.

**v2 had a circularity here** — `clinical_final_fit.ipynb` was defined as
"no folds," but a calibrator needs out-of-fold predictions, which only exist
inside the CV loop. Resolution:

1. `clinical_model_comparison.ipynb` writes every fold's held-out predictions to
   `results/clinical/oof_predictions.csv` as it runs. This costs nothing — the
   predictions already exist in memory.
2. `clinical_final_fit.ipynb` fits the model + MICE imputer on all 590 patients
   (no folds — validation is finished), then fits the calibrator on the OOF
   predictions from step 1, and saves all three artifacts.

Same pattern on the imaging side: `train_segmentation_detection.ipynb` writes
per-fold held-out predictions to `results/imaging/oof_predictions.csv`;
`imaging_evaluation.ipynb` fits the calibrator from those.

`fusion.ipynb` and `app.py` **load** calibrators. They never fit one.

---

## 7. Git policy for the new folders

Add to `.gitignore`: nothing new for `outputs/` or `results/` — **both should be
committed.** They are small (PNGs and CSVs), they are what Chapters 4–5 are
written from, and losing them means re-running training to recover a number.

`checkpoints/` stays ignored — large binaries, reproducible from the notebooks.
`data/` stays ignored. `_archive/` is committed (it's a notebook, and its whole
purpose is provenance).

---

## 8. How to extend this structure without redesigning it

This is written to absorb changes, since the project is still moving.

- **Adding a third tabular model?** Add it to the loop in
  `clinical_model_comparison.ipynb`. A new row appears in `model_comparison.csv`.
  Nothing else changes — no new folders, no new files.
- **Adding a third imaging candidate?** New subfolder under
  `checkpoints/imaging/candidates/<name>/`. Nothing else changes.
- **A different model wins than expected?** Nothing changes. `final/model.pkl`
  and `final/model.pt` are generic; `model_card.json` records the identity.
  `app.py` and `fusion.ipynb` are untouched.
- **Adding a third dataset (e.g. another CT source)?** `data/raw/<name>/`,
  `data/processed/images/<name>/`, a new `dataset` column value in `manifest.csv`.
  The manifest already carries the MSD/NIH distinction, so this pattern is proven.
- **Fusion ever becomes a *learned* step** (meta-learner rather than a fixed
  rule)? Then and only then create `checkpoints/fusion/final/`. It does not exist
  now because a fixed combination rule is not learned and has nothing to save —
  and per Section 2, there's no paired data to fit a meta-learner on anyway.
- **Ran out of time for a piece?** Every notebook is independently runnable from
  saved inputs, so an unfinished branch never blocks a finished one.

**`model_card.json` shape** (keep it trivial — it exists so the dashboard and
report never guess):

```json
{
  "model": "xgboost",
  "trained_on": "2026-07-15",
  "n_patients": 590,
  "imputer": "MICE_CA19_9Imputer",
  "cv_auc_repeated_5x10": 0.908,
  "cv_early_recall": 0.722,
  "notes": "winner of clinical_model_comparison.ipynb"
}
```

---

## 9. Migration instructions for Claude Code

**Working preference for this project: explain before you build, show diffs, do
not auto-accept. Steps 8–12 are scaffolding only — structure, imports, path
config, fold handling. Do not write full modeling implementations unless asked.**

1. Do not touch `data/raw/`. Do not delete anything in `checkpoints/` or `data/`.
2. Create `outputs/eda/{ct,tabular}/`, `outputs/qa/{ct,tabular}/`, and
   `outputs/eval/{imaging,clinical,fusion}/`. Create
   `results/{clinical,imaging,fusion}/`.
3. Move `qa_preprocessed_overlay.png` and `qa_nih_preprocessed.png` from
   `data/processed/` to `outputs/qa/ct/` — they are QA figures, not pipeline
   inputs. Update `imaging.ipynb`'s savefig paths accordingly. **This is the only
   change to `imaging.ipynb`; its preprocessing logic is done and verified — do
   not rewrite it.**
4. Update `CT_EDA.ipynb` and `Tabular_EDA.ipynb` to `savefig` their key figures
   into `outputs/eda/ct/` and `outputs/eda/tabular/`. Check each notebook's
   existing cells first — some figures may already be saved; do not assume none are.
5. Put all paths in `src/utils/config.py` — every new notebook imports from there
   rather than hardcoding `C:\FYP\...`. This is what makes the tree movable.
6. Populate `docs/` with every file in Section 3, including this migration file.
7. Create `src/_archive/`. Split `clinical.ipynb` per Section 5:
   - `tabular_clean.ipynb` — fixed-rule cleaning only (label mapping,
     sentinel-fill, REG1A drop) -> `data/processed/tabular_clean.csv`.
     **CA19-9 stays raw NaN.**
   - `clinical_imputer_benchmark.ipynb` — the existing KNN-vs-MICE comparison,
     moved near-verbatim; also writes `results/clinical/imputer_benchmark.csv`.
   - `clinical_model_comparison.ipynb` — loads `tabular_clean.csv`, MICE only,
     XGB/LogReg/RF across all three CV schemes, writes `model_comparison.csv`
     **and `oof_predictions.csv`**.
   - Then move the original to `src/_archive/clinical.ipynb`.
8. **Verification gate:** re-run `clinical_model_comparison.ipynb` and confirm
   XGBoost still matches Section 1 (AUC 0.908+/-0.023 repeated CV, 0.908
   cohort-out, 0.820+/-0.032 leave-one-site-out). **If the numbers do not match,
   stop and report it — do not proceed to the final fit.** A mismatch means the
   split broke something.
9. Create `clinical_final_fit.ipynb` — all-data fit (no folds) of winning model +
   MICE imputer, calibrator fit on `oof_predictions.csv` per Section 6, all three
   saved to `checkpoints/clinical/final/` with `model_card.json`.
10. Create `src/imaging/train_segmentation_detection.ipynb` — scaffolded to train
    both candidates using the manifest's **k=5 fold assignment** (do not assume a
    fixed train/val/test split exists), saving per-fold checkpoints to
    `checkpoints/imaging/candidates/<name>/fold_k.pt` and held-out predictions to
    `results/imaging/oof_predictions.csv`. Make the fold(s) to run configurable —
    on a 4GB card, full 5-fold on both candidates may not fit the schedule, so
    running a single fold must be a one-line change.
11. Create `src/imaging/imaging_evaluation.ipynb` — scaffolded to compare
    candidates, write `results/imaging/model_comparison.csv`, save figures to
    `outputs/eval/imaging/`, promote the winner + fitted calibrator to
    `checkpoints/imaging/final/` with `model_card.json`, and run the
    Grad-CAM/segmentation-overlap confound check (MSD and NIH come from different
    institutions — the detection head could be reading scanner signature rather
    than pathology; this check is the probe for that, and the result gets reported
    honestly either way).
12. Create `src/fusion/fusion_evaluation.ipynb` — scaffolded to read both
    branches' results and write `results/fusion/pair_comparison.csv`:
    one row per pair, each branch's own metrics as separate columns
    (`imaging_model, imaging_auc, tabular_model, tabular_auc, ...`).
    **No joint metric column.**
13. Update `.gitignore` per Section 7 — `outputs/` and `results/` are committed.
14. Show a diff/summary of every file created, moved, or modified before
    finalizing. Do not delete `PDAC_FYP_Folder_Structure_Proposal.md` — it moves
    to `docs/` as history.
