# PDAC FYP — Project Handoff / Continuation Context

**Purpose of this file:** paste this into a new chat to continue exactly where this one left off. It contains the project's technical decisions, what's been reviewed, what's been decided, and what's next — including things that deliberately deviate from the written IR.

**Student:** Aley Ahmed Nabil Elboraie (TP075961) — APD3F2511CS(DA)
**Today's context when this was written:** 4 days into a 20-day plan, 3 days behind schedule, compressed to a 17-day catch-up plan (4–20 Jul 2026), hard submission deadline 22 Jul 2026.

---

## 1. How I want to work (read this first)

I don't want code just handed to me. Before writing code, explain in plain language what you're about to build and why (3–5 sentences). After writing it, explain what non-trivial parts of the code actually do. At the end of a work session, quiz me on what we built — if I can't answer, slow down before moving on. I'd rather lose time to understanding than finish fast and not be able to defend my own project.

## 2. System architecture (confirmed, locked)

Multimodal ML framework for early PDAC (pancreatic ductal adenocarcinoma) detection, fusing CT imaging and urinary biomarker data into one interpretable risk score.

| Component | Spec | Why |
|---|---|---|
| **Imaging branch** | 2D U-Net, pretrained ResNet-50 encoder, multitask (segmentation mask + slice-level risk score). Dataset: MSD Task07 Pancreas CT (281 volumes, ~26,719 slices). | 2D not 3D due to RTX 3050's 4GB VRAM limit. ResNet-50 pretrained on ImageNet gives stronger low-level features on a small medical dataset than training an encoder from scratch. Multitask (shared encoder, two heads) is literature-supported to improve both tasks vs. training separately. |
| **Clinical branch** | XGBoost on Debernardi et al. (2020) urinary biomarker dataset (590 patients, 3 classes: Control/Benign/PDAC). SMOTE for class imbalance. | Neural nets overfit on 590 rows; XGBoost handles small tabular data better, natively handles missing values, integrates cleanly with SHAP. |
| **Fusion** | Late fusion — combine each branch's output score, not raw features. | Justified by dataset scale/modularity; the two datasets aren't paired per-patient, so early/joint fusion isn't viable. |
| **Explainability** | Grad-CAM (imaging — which pixels), SHAP (clinical — which features). | Different data types need different explanation methods; both feed into the dashboard, not just notebooks. |
| **Deployment** | Streamlit dashboard: CT upload → segmentation + score; tabular upload → XGBoost prediction; combined view → fusion score + Grad-CAM overlay + SHAP chart. | — |
| **Hardware** | Intel i7-12700H, RTX 3050 (4GB VRAM), 16GB RAM, SSD. Cloud fallback: Google Colab for heavier training. | Drives the 2D-not-3D decision above. |

## 3. Deviations from the written IR (deliberate — documentation needs to catch up later)

These are decisions made in this chat that the IR doesn't yet reflect. When rewriting documentation, these need updating:

1. **Python 3.13, not 3.10.** IR Chapter 2.4 states Python 3.10. Working environment actually uses 3.13 (confirmed via notebook kernel metadata). Decision: stay on 3.13, deliberately.
2. **PyRadiomics dropped, replaced with scikit-image.** PyRadiomics has no Python 3.13 wheel (verified directly against PyPI — last prebuilt wheels were for 3.7–3.9) and its source build fails on 3.10+ per upstream GitHub issues (AIM-Harvard/pyradiomics #903, #932). Replacement: a hand-rolled feature set using `skimage.measure.regionprops` (shape), NumPy/SciPy on masked voxels (first-order intensity), and `skimage.feature.graycomatrix`/`graycoprops` (GLCM texture) — covers the same three feature families PyRadiomics provided, minus the extra filtered-image derivatives (LoG, wavelet) and less-common texture matrices (GLRLM, GLSZM, NGTDM, GLDM), which aren't needed for the project's core objectives. **Table 7 in IR Chapter 2.4 needs rewriting to justify this swap.**
3. **Streamlit isn't in the IR's tech stack table at all** (Chapter 2.4, Tables 4–12) — it was only ever mentioned in the Notion 20-day plan. Needs a justification entry added to Chapter 2.4 documentation.
4. General principle going forward: **the IR is editable, not gospel.** If something has a better/faster/more-compatible alternative, recommend it — just flag what documentation needs to change as a result.

## 4. Environment setup status — DONE (2026-07-04)

- No conda on this machine — used `venv` instead, same isolation, kernel still named `fyp_env`. Env lives at `C:\FYP\fyp_env`.
- PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 installed (driver supports CUDA 13.2; cu128 chosen over cu130 — cu130's wheel index has an open packaging bug missing `cuda-bindings`). `torch.cuda.is_available()` confirmed `True`.
- `requirements.txt` installed clean, zero conflicts. `requirements-lock.txt` frozen (168 pkgs).
- Smoke test passed. One doc bug fixed: `SETUP.md` said `import grad_cam`, actual module is `pytorch_grad_cam` (pip name `grad-cam` != import name).
- **Still open:** `wandb login` — interactive, needs the student to paste their API key themselves.

## 5. Folder structure (finalized)

```
C:\FYP\
├── data\
│   ├── raw\              # untouched MSD Task07 + Debernardi CSV
│   └── processed\        # resampled CT slices, cleaned tabular CSV
├── src\
│   ├── imaging\          # U-Net + ResNet-50, preprocessing, Grad-CAM, radiomics-lite features
│   ├── clinical\         # XGBoost, SMOTE pipeline, SHAP
│   ├── fusion\           # late fusion logic
│   └── utils\            # shared helpers (metrics, config)
├── notebooks\
│   ├── Tabular_EDA.ipynb
│   └── CT_EDA.ipynb      # not yet reviewed — see Section 7
├── dashboard\            # streamlit app
├── checkpoints\          # saved model weights (gitignored)
├── docs\                 # chapter drafts as markdown before final Word export
└── requirements.txt
```

Rationale: each branch (`src/imaging`, `src/clinical`, `src/fusion`) is independent until fusion — different datasets, different libraries, trained separately — so the folders mirror that. `notebooks/` is exploration-only; reusable logic belongs in `src/`, not trapped in notebook cell order.

## 6. Catch-up schedule (compressed, 17 days: 4–20 Jul 2026, buffer to 22 Jul)

| New Day | Date | Focus |
|---|---|---|
| 1 | Fri 4 Jul | Setup + CT data loading |
| 2 | Sat 5 Jul | U-Net (ResNet-50) training |
| 3 | Sun 6 Jul | Tabular data + XGBoost |
| 4 | Mon 7 Jul | Late fusion |
| 5 | Tue 8 Jul | Streamlit dashboard |
| 6 | Wed 9 Jul | End-to-end testing & debugging |
| 7–12 | Thu 10–Tue 15 Jul | Documentation, Chapters 4–6, 1–3, Abstract, References |
| 13 | Wed 16 Jul | Poster design |
| 14 | Thu 17 Jul | Demo video |
| 15 | Fri 18 Jul | Full documentation edit pass |
| 16 | Sat 19 Jul | Final review |
| 17 | Sun 20 Jul | Submission |
| — | 21–22 Jul | Buffer |

**Note:** the redo of the EDA/preprocessing work (Section 7 below) predates Day 1's coding work and needs to land before the compressed schedule's Day 1 tasks start in earnest, since Day 2's U-Net training and Day 1's CT loading depend on the preprocessing pipeline being solid.

## 7. EDA review status

### Tabular EDA (`Tabular_EDA.ipynb`) — reviewed, verdict: solid foundation, not a rebuild

**What's already good:** dataset overview, class distribution (with cohort + sample origin cross-tabs), missing value analysis (including % missing by diagnosis class), demographic analysis, biomarker distributions (raw/log/boxplot), CA19-9-specific analysis with clinical cutoff (37 U/ml), correlation heatmap, PDAC-only stage analysis.

**Gaps identified, to add (not a rebuild):**
1. **Skewness incomplete** — notebook only computes skewness for LYVE1/REG1B/TFF1/REG1A, never for `plasma_CA19_9` or `creatinine`. IR's own headline claim (skewness reaching 10.37 in plasma_CA19_9) isn't backed by the notebook's actual output — needs adding before writing the preprocessing chapter.
2. **No quantified outlier detection** — boxplots show outliers visually but nothing computes IQR bounds or counts per biomarker per class. IR promises "IQR-based outlier detection with biologically justified retention"; need real numbers to write that honestly.
3. **MNAR claim not actually tested** — missingness % by diagnosis class is shown, but MNAR specifically means missingness relates to the value itself or an unmeasured cause. Should check missingness against `sample_origin`/`patient_cohort` too — if certain sites simply never ran the CA19-9 assay, that's missing-by-protocol, not MNAR in the clinical sense, and changes the imputation justification.
4. **Correlation-with-diagnosis uses Pearson against a nominal 3-class label** — worth supplementing with Kruskal-Wallis (biomarkers are skewed) to make the "strongest predictors" claim defensible if an examiner pushes on it.
5. **No data-quality sanity checks** — no duplicate `sample_id` check, no categorical value validation.
6. **Two real findings currently undiscussed:** Cohort1 has 162 PDAC vs Cohort2's 37 (matters if ever splitting by cohort); UCL sample origin contributes only Benign cases (0 Control, 0 PDAC) — a representativeness caveat.
7. **Structurally-missing vs. actually-missing not distinguished** — `stage` and `benign_sample_diagnosis` are missing by design for non-applicable diagnosis classes; currently lumped in with true missingness (plasma_CA19_9, REG1A), which will read as sloppy if not separated in the writeup.

**Decision on priority:** items 1–3 need actual code additions (IR already promises these exist). Items 4–7 are documentation-only fixes — the numbers already exist in the notebook's printed output, just need proper write-up.

**Not yet started:** writing the actual code additions for items 1–3.

### CT EDA notebook — not yet shared/reviewed

Next step when picking this up.

## 8. Files already produced (in `/mnt/user-data/outputs/` in the prior chat)

- `CLAUDE_CODE_BRIEF.md` — full project brief + day-by-day Claude Code prompts for the compressed schedule
- `requirements.txt` — verified Python 3.13-compatible dependency list, grouped by IR's own library tables

## 9. Immediate next steps (pick up here)

**Repo scaffolding — DONE (2026-07-04):** folder structure built per Section 5. Raw data moved in: `data/raw/Task07_Pancreas/` (MSD imagesTr/imagesTs/labelsTr, ~12GB) and `data/raw/Debernardi et al 2020 data.csv`. macOS AppleDouble junk (`._*`, `.DS_Store`) stripped during the move. `.gitignore` added (`fyp_env/`, `checkpoints/`, `data/`, `wandb/`, etc.) — not yet a git repo, gitignore is prepped for whenever `git init` happens.

**Gap found:** `Tabular_EDA.ipynb` and `CT_EDA.ipynb`, referenced throughout Section 7 as already-reviewed, do not exist anywhere on this machine — `notebooks/` is currently empty. Either they exist only on another machine/prior environment and need to be copied over, or they need to be rebuilt. Resolve this before EDA work in step 1-2 below can actually proceed.

1. Locate or rebuild the CT EDA and Tabular EDA notebooks (see gap above), then share CT EDA for review (same treatment as tabular: what's solid, what's missing).
2. Write the three priority additions to the tabular EDA (skewness for all biomarkers, IQR outlier quantification, missingness-by-cohort/origin check) — with explanation-first, not code-dump.
3. Once both EDAs are solid, move to the preprocessing pipelines proper (tabular pipeline is likely closer to done per IR; CT pipeline is flagged in the IR itself as incomplete — 2D slice extraction and 3-channel replication for ResNet-50 compatibility are explicitly marked "outstanding," and SMOTE/weighted sampling for class imbalance are "to be finalized").
4. Then begin Day 1 of the compressed schedule proper: CT data loading pipeline (env + scaffolding are no longer blockers).
