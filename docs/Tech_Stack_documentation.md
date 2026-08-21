# Technology Stack — PDAC Multimodal Risk Detection FYP

This document catalogues the hardware, software, languages, frameworks, IDEs, libraries,
and tools actually used across this project, as evidenced by the codebase
(`requirements.txt`, `requirements-lock.txt`, `.vscode/settings.json`,
`.streamlit/config.toml`, `.claude/launch.json`, source/notebook imports, and
`PDAC_FYP_Project_Handoff.md`). Version numbers are taken from `requirements-lock.txt`
(the verified, pinned-after-install record), not the floor versions in
`requirements.txt`.

---

## 1. Hardware

| Role | Spec | Notes |
|---|---|---|
| **Local development machine** | Intel i7-12700H, RTX 3050 (4GB VRAM), 16GB RAM, SSD, Windows 11 | Used for CT/tabular preprocessing, EDA, dashboard development, and all non-training work. 4GB VRAM was the reason the imaging model is 2D rather than 3D. |
| **Cloud training GPU (rented)** | RunPod on-demand pod: **RTX 6000 Ada, 48GB VRAM** (~$0.77/hr measured) | Local RTX 3050 was measured at 37.81 min/epoch and confirmed compute-bound (97% GPU utilization), which was too slow for the project schedule. All imaging model training (3-fold candidate run, 5-fold candidate run, two confound-check retrain rounds, final all-data fit) ran on this rented pod instead. Total spend across all imaging training: ~$3-4. Pods are spun up per run and terminated afterward (no persistent cloud state). |

---

## 2. Programming Language and Framework Chosen

| Category | Choice | Notes |
|---|---|---|
| **Programming language** | **Python 3.13** | `requirements.txt` explicitly notes this deviates from an earlier-stated Python 3.10 in the IR/report; 3.13 was chosen because all required packages had verified wheels for it. Confirmed via `__pycache__/*.cpython-313.pyc` and `.vscode/settings.json`'s interpreter path. |
| **Deep learning framework** | **PyTorch** (`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`) + **MONAI** (`monai==1.6.0`) | PyTorch is the core tensor/autograd/training framework; `torchvision` supplies the pretrained ResNet-50 encoder backbone. MONAI supplies medical-imaging-specific losses, transforms, and sliding-window inference utilities on top of PyTorch. |
| **Classical ML framework** | **XGBoost** (`xgboost==3.3.0`), with **scikit-learn** (`scikit-learn==1.9.0`) for pipelines/splitting/metrics and **imbalanced-learn** (`imbalanced-learn==0.14.2`) for SMOTE | XGBoost is the clinical/tabular branch's classifier, chosen over Logistic Regression and Random Forest after a documented 3-way comparison (`src/clinical/clinical_model_comparison.ipynb`). |

---

## 3. Integrated Development Environment

| Tool | Evidence |
|---|---|
| **Visual Studio Code** | `.vscode/settings.json` sets `python.defaultInterpreterPath` and `python.analysis.extraPaths`. |
| **Jupyter / JupyterLab** (`jupyter==1.1.1`, `jupyterlab==4.6.1`, `ipykernel==7.3.0`, `notebook==7.6.0`) | Nearly all modeling work (imaging training, clinical fitting/SHAP, fusion, EDA) is done in `.ipynb` notebooks under `src/` and `notebooks/`, run via VS Code's notebook interface or JupyterLab. |
| **RunPod web console / SSH** | Used to provision, run training on, and terminate the rented cloud GPU pod (PyTorch pre-installed via the pod template). |

---

## 4. Libraries and Tools

### Deep learning and model development
- **torch** 2.11.0 (+cu128) — core DL framework, autograd, training loop
- **torchvision** 0.26.0 (+cu128) — pretrained ResNet-50 encoder
- **monai** 1.6.0 — medical-specific losses, transforms, sliding-window inference

### Medical imaging processing
- **SimpleITK** 2.5.5 — CT resampling to uniform voxel spacing
- **nibabel** 5.4.2 — loading/reading NIfTI (`.nii`/`.nii.gz`) CT volumes
- **pydicom** 3.0.2 — DICOM reading (for conversion to NIfTI where needed)
- **opencv-python** (cv2) 5.0.0.93 — 2D slice resizing/processing
- **scikit-image** 0.26.0 — hand-rolled shape (`regionprops`), first-order, and GLCM texture (`graycomatrix`/`graycoprops`) features (chosen as a replacement for PyRadiomics, which has no Python 3.13 wheel)

### Classical machine learning
- **xgboost** 3.3.0 — clinical branch risk classifier
- **scikit-learn** 1.9.0 — splitting, preprocessing pipelines, cross-validation, metrics
- **imbalanced-learn** 0.14.2 — SMOTE for clinical dataset class imbalance

### Explainable AI (XAI)
- **grad-cam** (pytorch-grad-cam) 1.5.5 — Grad-CAM heatmaps for the imaging branch
- **shap** 0.52.0 — feature-level explanation of XGBoost predictions (clinical branch), plotted via `shap.TreeExplainer`
- **captum** 0.9.0 — Integrated Gradients, used to cross-validate Grad-CAM
- **lime** 0.2.0.1 — model-agnostic explanations, used to cross-validate SHAP

### Data handling and analysis
- **numpy** 2.4.4
- **pandas** 3.0.3
- **scipy** 1.18.0 — statistical tests during EDA (t-tests, non-parametric tests)
- **statsmodels** — McNemar test, XGBoost-vs-Random Forest comparison in `clinical_model_comparison.ipynb`

### Visualisation
- **matplotlib** 3.11.0 — training curves, CT/mask visualisation, XAI figures
- **seaborn** 0.13.2 — statistical EDA plots, correlation heatmaps
- **plotly** — interactive charts on the Streamlit dashboard (Predict/Analytics pages)
- **altair** 6.2.2 — bundled with Streamlit for chart rendering

### Experiment tracking
- **wandb** (Weights & Biases) 0.28.0 — logging training metrics, hyperparameters, and run comparisons for imaging model training

### Deployment (not in original IR tech-stack list — used for the dashboard deliverable)
- **streamlit** 1.58.0 — the dashboard app itself (`dashboard/`: Register → Predict → Analytics → Reports → About)
- **reportlab** — PDF report export on the dashboard's Reports page
- **pypdf** — strips a trailing blank page from ReportLab's auto-pagination
- **plotly** — interactive dashboard charts (also listed under Visualisation above)

### Operating System
- **Windows 11** — both the local development machine and (per the handoff doc) the target OS; the RunPod training pod ran a Linux-based container image but is otherwise not the primary development OS.

---

## 5. Categories not explicitly requested but worth documenting

- **Project/version control:** **Git**, hosted on **GitHub** (`origin` → `github.com/El-boraie/turbo-engine-fyp`).
- **Environment/package management:** **conda** (env name `fyp_env`, Python 3.13) for environment creation, **pip** for package installation, with `requirements.txt` (floor versions) and `requirements-lock.txt` (`pip freeze` snapshot) as the reproducibility record.
- **Cloud GPU rental platform:** **RunPod** — on-demand GPU pod provisioning for imaging model training (see Hardware section).
- **Datasets used:** MSD Task07 Pancreas CT, NIH Pancreas-CT (imaging), Debernardi et al. (2020) urinary biomarker dataset (clinical/tabular) — sourced via `notebooks/download data.py`; PANORAMA is referenced in `docs/PANORAMA_Dataset_Summary_Table.md` as a dataset considered/summarized.
