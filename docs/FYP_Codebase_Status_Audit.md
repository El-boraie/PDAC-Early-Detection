# PDAC FYP — Codebase Status Audit

**What this is:** a snapshot check of what in the repo is real, working, executed logic versus scaffolding (structure only, no modeling implementation), verified by directly inspecting notebook cell contents and execution outputs — not by assuming the status quoted in `FYP_Folder_Structure_Manual.md` is still accurate. Written to confirm that manual against the actual repo state.

**Audit date:** 2026-07-15.

**Method:** every notebook in `src/`, `notebooks/`, and `src/_archive/` was parsed programmatically (via `nbformat`) and checked for: code cell count, presence of execution outputs, presence of error outputs, count of `raise NotImplementedError` stub cells, and presence of real logic call-sites (`.fit(`, `.predict(`, `predict_proba(`, `torch.save`, `np.save(`, `to_csv(`, `savefig(`, `joblib.dump(`). `config.py`/`metrics.py` were read directly.

---

## Full working logic (executed, real code, no stubs)

| File | What it does |
|---|---|
| `notebooks/CT EDA.ipynb` | CT exploratory analysis — 22 code cells, all real, executed clean, no errors |
| `notebooks/Tabular EDA.ipynb` | Biomarker exploratory analysis — 39 code cells, all real, executed clean |
| `src/imaging/imaging.ipynb` | Full CT preprocessing (MSD + NIH) — real `np.save`/`to_csv`/`savefig` calls throughout, executed clean |
| `src/clinical/tabular_clean.ipynb` | Fixed-rule tabular cleaning — writes `tabular_clean.csv`, executed clean |
| `src/clinical/clinical_imputer_benchmark.ipynb` | KNN-vs-MICE comparison — real `.fit()`/`.predict()` calls throughout both imputer classes and the CV loop, executed clean |
| `src/_archive/clinical.ipynb` | The original archived notebook — same real logic as the imputer benchmark, preserved as-is |
| `src/utils/config.py` | Not code logic per se, but complete — every path constant is real and in use |
| `src/utils/metrics.py` | **Partially real** — see split below |

## Scaffolding (structure only — imports, config, manifest-reads work; every actual modeling cell raises `NotImplementedError`)

| File | Real cells | Stub cells |
|---|---|---|
| `src/imaging/train_segmentation_detection.ipynb` | imports, `CANDIDATES`/`FOLDS_TO_RUN` config, manifest load-and-print (3 cells, all execute correctly) | training loop (1 cell, `raise NotImplementedError`) |
| `src/imaging/imaging_evaluation.ipynb` | imports (1 cell) | candidate comparison, promotion, Grad-CAM confound check (3 cells, all `raise NotImplementedError`) |
| `src/fusion/fusion_evaluation.ipynb` | imports (1 cell) | loading both branches' results, pair-comparison table (2 cells, both `raise NotImplementedError`) |
| `src/utils/metrics.py` — `dice_score`/`iou_score` | — | both raise `NotImplementedError` immediately, no segmentation model exists yet to validate against |
| `src/utils/metrics.py` — `pr_auc`/`early_stage_recall` | real, working functions ported from the verified clinical logic | — |

## Empty / not built at all

- `src/fusion/fusion.ipynb`, `src/utils/utils.ipynb` — 0 cells, untouched pre-existing placeholders
- `dashboard/app.py` — doesn't exist
- `checkpoints/**` — every subfolder present but empty (nothing trained)
- `src/clinical/clinical_model_comparison.ipynb`, `clinical_final_fit.ipynb` — confirmed absent (rolled back same day as the migration — see `docs/FYP_Migration_Changelog.md`)
- `results/imaging/`, `results/fusion/` — empty except `.gitkeep`

---

## Alignment with `FYP_Folder_Structure_Manual.md`

**Fully aligned, no discrepancies found.** Every status the manual claims — `DONE+VERIFIED`, `STUB`, `PARTIAL`, `NOT YET CREATED`, `N/A` — matches what's actually on disk, down to specific cell counts and file lists:

- `outputs/eda/ct/` really does have exactly 10 files, `outputs/eda/tabular/` exactly 25.
- `metrics.py` really is split exactly as described: clinical functions real, imaging functions stubbed.
- The manual's description of the rollback state (`clinical_model_comparison.ipynb`/`clinical_final_fit.ipynb` as "NOT YET CREATED (after rollback)") is accurate — confirmed neither exists anywhere on disk, and nothing else references them except historical prose mentions in `clinical_imputer_benchmark.ipynb`, `imaging_evaluation.ipynb`, `train_segmentation_detection.ipynb`, and `metrics.py` (name-only mentions in markdown/comments, not imports or dependencies).

Nothing has drifted since the manual was written.
