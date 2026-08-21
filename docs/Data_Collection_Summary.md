# Data Collection Summary — All Datasets Used in This Project

One reference document covering every dataset this project touches: what it
is, where it came from, its license, and exactly how it was used (training,
healthy-negative addition, external stress test, or clinical/tabular
branch). Each dataset's table follows the same property structure as
Tables 15–17 already in the IR, so this can be dropped in as one continuous
block or split back into individual tables.

**Datasets covered:** (1) MSD Task07 Pancreas, (2) NIH Pancreas-CT, (3)
PANORAMA Challenge, (4) Debernardi et al. urinary biomarker dataset.

---

## 1. MSD Task07 — Pancreas (Primary Imaging Dataset)

| Property | MSD Task07 (Primary) |
|---|---|
| Source Institution | Memorial Sloan Kettering Cancer Center |
| Published Reference | Antonelli, M., Reinke, A., Bakas, S., et al. (2022). *The Medical Segmentation Decathlon*. Nature Communications, 13(1), 4128. https://doi.org/10.1038/s41467-022-30695-9 |
| Download Link | http://medicaldecathlon.com/ (official portal) — also mirrored on the AWS Registry of Open Data: https://registry.opendata.aws/msd/ (`aws s3 ls --no-sign-request s3://msd-for-monai/`) |
| Total 3D Volumes | 420 (281 training, with pancreas + tumour masks; 139 test, unlabelled — not used in this project) |
| Estimated 2D Slices (Train) | ~26,719 |
| Annotation Type | Pancreas + Tumour (voxel-level) |
| Modality | Portal venous phase CT |
| Format | NIfTI (`.nii.gz`) |
| Subject Condition | PDAC confirmed |
| License | CC BY-SA 4.0 |
| Role in this project | Primary training data for the imaging branch — 281 training patients, both segmentation (masked to annotated slices) and detection heads |
| Local path | `data/raw/Task07_Pancreas/` (`imagesTr/`, `labelsTr/`, `dataset.json`) |

*Verified locally against `data/raw/Task07_Pancreas/dataset.json`, which
records `licence: CC-BY-SA 4.0`, `reference: Memorial Sloan Kettering Cancer
Center`, `numTraining: 281`, `numTest: 139` directly — this is the dataset's
own bundled metadata file, not an inferred value.*

---

## 2. NIH Pancreas-CT (Healthy Negative Cohort)

| Property | NIH Pancreas-CT (Healthy Negative Cohort) |
|---|---|
| Source Institution | National Institutes of Health (NIH) Clinical Center, Bethesda, MD, USA — accessed in this project via a Hugging Face mirror (`CADS-dataset/0024_pancreas_ct`) |
| Published Reference | Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). *Data From Pancreas-CT* (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU |
| Download Link | https://www.cancerimagingarchive.net/collection/pancreas-ct/ (official TCIA collection) — this project's actual copy: https://huggingface.co/datasets/CADS-dataset/0024_pancreas_ct (pre-converted NIfTI mirror) |
| Total 3D Volumes | 82 scans from 80 subjects (80/80 loaded and used in this project) |
| Estimated 2D Slices | ~18,942 |
| Annotation Type | Pancreas (manual, by a medical student, radiologist-verified) — **in the original TCIA release only**. Not used here: the Hugging Face mirror shipped a different, multi-organ auto-segmentation mask with no label-to-organ mapping, judged unusable, so this pass is image-only |
| Modality | Portal venous phase CT (contrast-enhanced, ~70s post-injection) |
| Format | NIfTI (`.nii.gz`) as received via the mirror — pre-converted; the original TCIA release format is DICOM |
| Subject Condition | Healthy controls (no pancreatic or abdominal abnormality) |
| License | CC BY 3.0 |
| Role in this project | Healthy-negative cohort for the imaging detection head — added specifically because MSD Task07 alone is 100% cancer-positive |
| Local path | `data/raw/NIH_Pancreas_CT/0024_pancreas_ct/` (`images/`, `segmentations/` (unused), `README_0024_pancreas_ct.md`) |

*Verified locally against the mirror's own bundled
`README_0024_pancreas_ct.md`. Also has its own standalone table:
`docs/NIH_Dataset_Summary_Table.md`, including a flagged inconsistency in the
source README's own volume count (80 vs. 82).*

---

## 3. PANORAMA Challenge (External, Out-of-Distribution Test Set)

| Property | PANORAMA Challenge |
|---|---|
| Source Institution | Multi-center: Radboud UMC & UMC Groningen (Netherlands, primary), plus Ziekenhuis Groep Twente (NL), Karolinska Institutet (Sweden), Haukeland University Hospital (Norway) |
| Published Reference | Alves, N., Schuurmans, M., Yakar, D., Vendittelli, P., Litjens, G., Hermans, J., & Huisman, H. (2024). *The PANORAMA Challenge: Public Training and Development Dataset (1)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13715870 |
| Download Link | https://panorama.grand-challenge.org/datasets-imaging-labels/ (challenge page) → https://zenodo.org/records/13715870 (Batch 1 of 4, the batch used here) |
| Total 3D Volumes | 557 volumes used in this project, from Batch 1 of the official 4-batch, 2,238-case release |
| Estimated 2D Slices | Not applicable — used whole-volume, never converted to a slice-level set |
| Annotation Type | Pancreatic tumour delineation — 108 expert/manual, 449 AI-derived/automatic (official `panorama_labels` set). Used only to derive a PDAC-positive/negative flag (158 positive, 399 negative) for reference — not used as training signal |
| Modality | Contrast-enhanced CT (CECT) |
| Format | NIfTI (`.nii.gz`) |
| Subject Condition | Mixed (PDAC-positive and non-PDAC) |
| License | CC BY-NC 4.0 (non-commercial) |
| Role in this project | **Never used for training or evaluation.** Out-of-distribution stress test for the dashboard's raw-upload pipeline only — genuinely unseen data, 5 of 557 volumes processed end-to-end with no crashes (scores 0.09–0.99); see `docs/Dashboard_documentation.md` Section 5.2 |
| Local path | `data/testing/PANORAMA CHALLENGE Batch 1/`, `data/testing/panorama_labels/`, `data/testing/panorama_labels_manifest.csv` |

*Full standalone table: `docs/PANORAMA_Dataset_Summary_Table.md`. Unlike MSD
and NIH, no bundled README exists locally for this one — citation/license/
batch facts were confirmed via live lookup against the official challenge
page and Zenodo record, not from a local file.*

---

## 4. Debernardi et al. Urinary Biomarker Dataset (Clinical / Tabular Branch)

| Property | Debernardi et al. Urinary Biomarker Dataset |
|---|---|
| Source Institution | Multi-center case–control study, led by Barts Cancer Institute, Queen Mary University of London (UK). Samples drawn from three coded biobank sites present directly in the data's `sample_origin` column: `BPTB` (Barts Pancreas Tissue Bank, UK), `ESP` (Spanish site, consistent with co-author affiliation at CNIO, Madrid), `LIV` (University of Liverpool, UK) |
| Published Reference | Debernardi, S., O'Brien, H., Algahmdi, A. S., et al. (2020). *A combination of urinary biomarker panel and PancRISK score for earlier detection of pancreatic cancer: A case–control study*. PLOS Medicine, 17(12), e1003489. https://doi.org/10.1371/journal.pmed.1003489 |
| Download Link | https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer (Kaggle mirror, the copy used in this project) — original article (with full supplementary data): https://doi.org/10.1371/journal.pmed.1003489 |
| Total Records | 590 patients, 3-class diagnosis (Control / Benign / PDAC), reframed as binary PDAC-vs-not for this project |
| Features Used | 7: `creatinine`, `LYVE1`, `REG1B`, `TFF1`, `plasma_CA19_9`, `age`, `sex` (an 8th column, `REG1A`, was dropped — 100% missing in one cohort) |
| Format | CSV (tabular) |
| Subject Condition | Mixed — Control (183), Benign (208), PDAC (199) |
| License | CC BY 4.0 (PLOS open-access license; applies to the underlying published dataset) |
| Role in this project | Sole data source for the clinical branch (XGBoost) — model comparison, final fit, and SHAP explainability all trained/evaluated on this dataset |
| Local path | `data/processed/tabular_clean.csv` (cleaned); raw source not stored in-repo |

*Citation/license verified via live lookup against the PLOS Medicine article
page and its stated CC BY 4.0 license statement. Biobank-site interpretation
(`BPTB`/`ESP`/`LIV`) is inferred from the codes actually present in the
dataset plus author affiliations in the paper — not an explicit site list
quoted from the paper itself, so treat that one line as reasoned inference,
not a direct quote.*

---

## Quick citation block (ready to paste)

```
Antonelli, M., Reinke, A., Bakas, S., et al. (2022). The Medical Segmentation
Decathlon. Nature Communications, 13(1), 4128.
https://doi.org/10.1038/s41467-022-30695-9

Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016).
Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive.
https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU

Alves, N., Schuurmans, M., Yakar, D., Vendittelli, P., Litjens, G., Hermans, J.,
& Huisman, H. (2024). The PANORAMA Challenge: Public Training and Development
Dataset (1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13715870

Debernardi, S., O'Brien, H., Algahmdi, A. S., et al. (2020). A combination of
urinary biomarker panel and PancRISK score for earlier detection of pancreatic
cancer: A case-control study. PLOS Medicine, 17(12), e1003489.
https://doi.org/10.1371/journal.pmed.1003489
```

## How these facts were verified

- **MSD** and **NIH**: confirmed against files already sitting in this
  project's own `data/raw/` folders (`dataset.json`, `README_0024_pancreas_ct.md`)
  — the strongest kind of verification, since it's the dataset's own bundled
  metadata, not a third-party restatement.
- **PANORAMA** and **Debernardi**: neither has a bundled README/license file
  in this project's local copy, so those facts were confirmed via live web
  lookups against the official challenge page, the Zenodo record, and the
  PLOS Medicine article page directly, done as part of producing this
  document — not recalled from memory. All source URLs are included above so
  they can be re-checked directly.
