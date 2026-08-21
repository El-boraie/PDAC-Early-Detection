# FYP Folder Structure Migration (v3) — Changelog

**What this is:** a record of every file created, moved, or modified while applying `docs/FYP_Folder_Structure_Migration.md` to the repo, what was verified along the way, and confirmation that nothing existing was lost. Plan approved before execution is preserved at the session's plan-mode checkpoint; this document is the after-the-fact record of what actually happened.

**Run date:** 2026-07-15. **Partially rolled back same day** — see "Rollback" section below; Phase 6 item 3 and Phase 7 as originally written describe work that was subsequently undone, kept here struck through for provenance rather than deleted from this record.

---

## Rollback (2026-07-15, same session)

The migration doc's step 7 authorized building `tabular_clean.ipynb` and `clinical_imputer_benchmark.ipynb` (preprocessing + CA19-9 imputation) — but its instruction to also build `clinical_model_comparison.ipynb` with real working logic went further than approved: that's modeling work (XGBoost/LogReg/RF comparison) the user hadn't asked to start yet. Flagged after the fact, and rolled back on request:

**Deleted:**
- `src/clinical/clinical_model_comparison.ipynb`
- `src/clinical/clinical_final_fit.ipynb` (depended entirely on the above)
- `results/clinical/model_comparison.csv`
- `results/clinical/oof_predictions.csv`

**Kept, unaffected:**
- `src/clinical/tabular_clean.ipynb`, `src/clinical/clinical_imputer_benchmark.ipynb`, `results/clinical/imputer_benchmark.csv` — this is the actual current stopping point.
- `src/_archive/clinical.ipynb` — still justified without `clinical_model_comparison.ipynb` existing, since `clinical_imputer_benchmark.ipynb` itself already reproduced the original notebook's repeated-CV/cohort-out/leave-one-site-out AUC numbers exactly.
- All scaffolding-only files (`train_segmentation_detection.ipynb`, `imaging_evaluation.ipynb`, `fusion_evaluation.ipynb`) and everything from Phases 1-5, 8-10 below — confirmed (not assumed) to contain no real training/comparison logic, only imports/config/stubs.

**Standing instruction going forward:** only build what's explicitly requested piece-by-piece. A planning/migration document describing a future step is a description of what's coming, not authorization to build it now — ambiguous scope gets a stop-and-ask, not a default-to-build.

---

## Discrepancies found between the migration doc and the actual repo

The migration doc's tree diagram assumed some things that didn't match what's really on disk. Each was resolved deliberately, not silently:

1. **EDA notebook filenames** — doc shows `CT_EDA.ipynb`/`Tabular_EDA.ipynb` (underscores); actual files are `notebooks/CT EDA.ipynb` / `notebooks/Tabular EDA.ipynb` (spaces). **Not renamed** — same principle the doc itself uses for the Debernardi CSV ("original filename kept, spaces and all").
2. **Root handoff doc** — doc expects `PDAC_FYP_Project_Handoff.md`; actual file is `PROJECT_HANDOFF.md`. **Kept as-is**, same document.
3. **Files that don't exist anywhere on this machine**: `CLAUDE_CODE_BRIEF.md`, `docs/PDAC_FYP_Folder_Structure_Proposal.md`, `docs/Tabular_EDA_Summary_and_Open_Issues.md`, `docs/Tabular_Everything_Else.md`, `docs/Tabular_Preprocessing_plan.md`, `docs/chapters/`. Same "only exists on another machine/Drive" gap `PROJECT_HANDOFF.md` already flagged once before, for the EDA notebooks. **Not fabricated.** If you have these on Drive, drop them into `docs/` — nothing in the codebase depends on them.
4. **`data/processed/images/` layout** — doc's tree shows `images/{msd,nih}/`; actual layout is `images/{train,val,test}/` (MSD, split-named from the original holdout) + `images/nih/` (90,693 files). **Not renamed/moved.** The 14 numbered migration steps never actually instruct this move, `manifest.csv`'s `dataset` column already disambiguates MSD vs. NIH per row, and renaming ~91k files would be slow, risky, and purely cosmetic against an illustrative diagram.
5. **Verification-gate cohort-out figure** (migration doc step 8) — its literal text lists "0.908 cohort-out" as the number `clinical_model_comparison.ipynb`'s XGBoost/MICE result should match. The real MICE cohort-out AUC is **0.8375** — 0.908 is KNN's cohort-out figure (and also the repeated-CV figure, for both imputers coincidentally close), and the doc's own Section 1 narrative ("KNN only edged ahead on cohort-out") confirms this. Read as a copy/paste slip in step 8, not a real target — verified against 0.8375 instead (see Verification Gate section below). Flagged in the new notebook's own markdown too, not just here.

---

## What changed, by phase

### Phase 1 — Folder scaffolding
Created: `outputs/eda/{ct,tabular}/`, `outputs/qa/{ct,tabular}/`, `outputs/eval/{imaging,clinical,fusion}/`, `results/{clinical,imaging,fusion}/`. `.gitignore` checked — already correct, no edit needed (`outputs/`/`results/` were never excluded; `data/`/`checkpoints/` still are, as intended).

### Phase 2 — QA images relocated, `imaging.ipynb` patched
- `qa_preprocessed_overlay.png` and `qa_nih_preprocessed.png` copied `data/processed/` → `outputs/qa/ct/`, verified byte-identical, then the originals removed (completing the "move" — no full notebook re-run needed since both PNGs were already correct, and the destination copy round-trip was verified first).
- `imaging.ipynb`: added `QA_OUTPUT_DIR` constant to the Configuration cell; both QA `savefig` calls (MSD overlay, NIH image-only) now write there. **This is the only change to `imaging.ipynb`** — its preprocessing logic (MSD + NIH pipeline, manifest, disk checks) is untouched.

### Phase 3 — EDA notebooks redirected + re-executed
- `Tabular EDA.ipynb`: one-line change, `OUT_DIR` repointed to `outputs/eda/tabular/`.
- `CT EDA.ipynb`: no `OUT_DIR` constant existed before this — added one, then prefixed all 8 `savefig` calls and 2 `to_csv` calls (they were writing bare filenames into the notebook's own working directory, which is why loose PNGs were scattered directly under `notebooks/`).
- Both re-executed successfully, zero errors. `outputs/eda/ct/` now has 10 files (8 PNGs + 2 CSVs), `outputs/eda/tabular/` has 25 files. Old scattered output files under `notebooks/` (loose PNGs, `notebooks/CT_eda_outputs/`, `notebooks/tabular_eda_outputs/`) were **left in place**, not deleted — harmless leftovers, a separate cleanup ask if wanted later.

### Phase 4 — `docs/` populated
`FYP_Folder_Structure_Migration.md` moved from repo root into `docs/` (per its own tree). The 6 missing files from discrepancy #3 were not fabricated.

### Phase 5 — `src/utils/config.py` + `src/utils/metrics.py` created
- `config.py`: every path the new notebooks need, centralized (raw data, `data/processed/`, `outputs/`/`results/`/`checkpoints/` subpaths, `RANDOM_SEED`), plus an `ensure_dirs()` helper. `clinical.ipynb`'s successors and the new imaging/fusion scaffolds import from here; `imaging.ipynb` itself keeps its own inline constants (per "do not rewrite").
- `metrics.py`: `EARLY_STAGES` and `early_stage_recall`/`pr_auc` ported as **real, working code** from the verified `clinical.ipynb` logic (needed by Phase 6's verification gate). `dice_score`/`iou_score` are signature-only stubs (`raise NotImplementedError`) — no segmentation model exists yet to validate them against.
- `src/utils/utils.ipynb` (pre-existing, empty) is untouched — not referenced anywhere in the migration doc.

### Phase 6 — `clinical.ipynb` split into three notebooks, verified, archived
1. **`tabular_clean.ipynb`** — Sections 1-6 of the original (load → targets → sentinel-fill+verify → drop `REG1A` → `FEATURES` → `METADATA`), plus a new final cell writing `data/processed/tabular_clean.csv` (590 rows × 14 columns: 5 metadata + 2 target + 7 feature columns; `plasma_CA19_9` stays raw `NaN`). Executed clean.
2. **`clinical_imputer_benchmark.ipynb`** — the original's Sections 7-9 (both imputer classes, `run_fold`, the 3-scheme KNN-vs-MICE comparison) moved near-verbatim, now loading `tabular_clean.csv` instead of re-deriving it. Writes `results/clinical/imputer_benchmark.csv`. Executed clean — **numbers matched the original run exactly** (repeated CV MICE AUC 0.9081±0.0229, cohort-out 0.8375, LOSO 0.8195±0.0322, all bit-identical to 4 decimals).
3. ~~**`clinical_model_comparison.ipynb`** — new logic: `MICE_CA19_9Imputer` only, extends the `run_fold` pattern to three models (XGBoost, `LogisticRegression` in an internally-scaled `Pipeline`, `RandomForestClassifier`) across the same three schemes. Writes `results/clinical/model_comparison.csv` (9 rows) and `results/clinical/oof_predictions.csv` (17,700 rows: 590 patients × 10 CV repeats × 3 models). Executed clean.~~
   **ROLLED BACK (2026-07-15, same session) — this went further than approved.** Building this notebook was authorized by the migration doc's step 7, but it's modeling work (comparing candidate models), not the preprocessing/imputation stage the user had actually approved. Deleted along with its two output CSVs — see "Rollback" section above. The findings below are kept for provenance only; they no longer exist on disk.
   - *(Design note the deleted notebook flagged in itself: `LogisticRegression` is scale-sensitive unlike XGBoost/RF, so it was wrapped as `Pipeline([StandardScaler(), LogisticRegression()])`, fit fresh per fold — mirroring the existing narrow exception in rule 2, where the KNN imputer's internal scaler never touches the shared feature matrix.)*
   - *(Notable finding at the time, not acted on before rollback: `RandomForest` edged out `XGBoost` on cohort-out (0.8901 vs. 0.8375) and leave-one-site-out (0.8324 vs. 0.8195); repeated CV was a near-tie (0.9076 vs. 0.9081). Worth re-deriving when this stage is actually reached.)*
4. **Verification gate — PASSED at the time**, via the now-deleted `clinical_model_comparison.ipynb`. XGBoost/MICE reproduced the pre-migration numbers exactly:

   | Scheme | Expected AUC | Actual AUC | Result |
   |---|---|---|---|
   | Repeated 5×10 CV | 0.9081 | 0.9081 | MATCH |
   | Cohort-out | 0.8375 | 0.8375 | MATCH |
   | Leave-one-site-out | 0.8195 | 0.8195 | MATCH |

   (Expected cohort-out figure is 0.8375, not the migration doc's literal "0.908" — see discrepancy #5.)

   **Post-rollback note:** this exact gate no longer exists on disk (it lived inside the deleted notebook), but the same three numbers are independently confirmed by the notebook that *does* still exist — `clinical_imputer_benchmark.ipynb`'s MICE row reports the identical repeated-CV/cohort-out/leave-one-site-out AUCs. That's why archiving `clinical.ipynb` (next item) remains justified without `clinical_model_comparison.ipynb` needing to exist.
5. Original `src/clinical/clinical.ipynb` moved to `src/_archive/clinical.ipynb`, unmodified, after the gate passed (still true post-rollback — see note above).

### Phase 7 — ~~`clinical_final_fit.ipynb` scaffolded~~ — ROLLED BACK

**This entire phase was undone in the same-day rollback** (see "Rollback" section above) — the notebook depended entirely on `clinical_model_comparison.ipynb`'s output and was itself past the approved stopping point. Kept here for provenance only; the file no longer exists.

*(What it contained, at the time: structure only, per "steps 8-12 are scaffolding only," except one real cell — reading `model_comparison.csv` and picking the winner by best repeated-CV AUC, which executed and correctly selected XGBoost, 0.9081, narrowly ahead of RandomForest's 0.9076. Everything downstream — all-data imputer+model fit, calibrator fit, checkpoint save — was a `TODO` + `raise NotImplementedError` stub; calibration method (Platt vs. isotonic) was never decided.)*

### Phase 8 — Imaging scaffolds
- **`train_segmentation_detection.ipynb`**: reads `manifest.csv`'s existing k=5 fold assignment (`split` column, `fold0`..`fold4` — confirmed present, not recomputed), `FOLDS_TO_RUN = [0]` as the one-line single-fold config point. Structural cells executed and confirmed correct (manifest read back: 90,693 rows, 281 MSD + 80 NIH patients). Training loop itself is a stub — no model architecture exists yet.
- **`imaging_evaluation.ipynb`**: structure for candidate comparison, promotion to `checkpoints/imaging/final/`, and the Grad-CAM/segmentation-overlap **confound check** — flagged in its own markdown as a real methodological concern (MSD and NIH are different institutions/scanners; the detection head could learn scanner signature instead of pathology) to actually resolve when this notebook is filled in, not a formality. All stubbed pending a trained model to evaluate.

### Phase 9 — `fusion_evaluation.ipynb` scaffolded
Structure for reading both branches' `model_comparison.csv` and writing `results/fusion/pair_comparison.csv` — one row per (imaging-model, tabular-model) pair, each branch's own metrics as separate columns, **explicitly no joint metric column** (no patient has both CT and urine data — migration doc Section 2). Stubbed; currently blocked on `imaging_evaluation.ipynb` producing a real `results/imaging/model_comparison.csv`.

### Phase 10 — this document, plus `.gitkeep` added to newly-created empty directories (`results/{imaging,fusion}/`, `outputs/eval/{imaging,clinical,fusion}/`, `outputs/qa/tabular/`) so they survive in git, matching the project's existing convention (`data/processed/.gitkeep`, `docs/.gitkeep`, etc.).

---

## Confirmation: nothing pre-existing was lost

- `checkpoints/` — untouched, still empty except `.gitkeep` (nothing trained yet; new empty subfolders `clinical/final/`, `imaging/candidates/`, `imaging/final/` created per the target tree, gitignored so no `.gitkeep` needed there).
- `data/raw/` — never touched.
- `data/processed/images/`, `masks/`, `manifest.csv`, `manifest_msd_only_backup.csv` — untouched.
- The only two pre-existing files removed from disk (`data/processed/qa_*.png`) were removed **only** after their copies at the new location were verified byte-identical — not a loss, a completed move.
- `src/clinical/clinical.ipynb` was **moved**, not deleted — it's at `src/_archive/clinical.ipynb`, byte-for-byte, and per the migration doc's own instruction stays there as provenance even after the split, not to be deleted at all in this pass.
- `src/imaging/imaging.ipynb`, `src/fusion/fusion.ipynb`, `src/utils/utils.ipynb` — all untouched except the 2 QA-path lines in `imaging.ipynb`.
- Old scattered EDA output files under `notebooks/` — left in place, not cleaned up.

**What *was* deleted** — not pre-existing work, but new-this-session work that overshot the approved scope, removed on request: `clinical_model_comparison.ipynb`, `clinical_final_fit.ipynb`, `results/clinical/model_comparison.csv`, `results/clinical/oof_predictions.csv`. See "Rollback" section above.

## Known follow-ups (not done in this pass, intentionally)

- `train_segmentation_detection.ipynb`, `imaging_evaluation.ipynb`, `fusion_evaluation.ipynb` all have real `TODO`-stub cells (`raise NotImplementedError`), confirmed by inspection to contain no working training/comparison logic — none of these run to completion yet, by design.
- `clinical_model_comparison.ipynb` and `clinical_final_fit.ipynb` no longer exist (rolled back) — to be rebuilt when that stage is explicitly approved, not before.
- The 6 doc files noted missing in discrepancy #3 are still missing.
- Old scattered EDA outputs under `notebooks/` are still there if you want them cleaned up separately.
